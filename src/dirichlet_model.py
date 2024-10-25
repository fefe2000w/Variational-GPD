'''
This module contains modified code from the sgpr.py module of GPFlow
https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py
referring to the heteroskedastic.py module
https://github.com/dmilios/dirichletGPC/blob/master/src/heteroskedastic.py

In particular, the DBModel class below is a modified version of SGPR
that offers an implementation of heteroskedastic GP regression
so that it admits a different vector of noise values for each output dimension.

Different from the SGPRh class in heteroskedastic.py module,
the DBModel class integrates hyperparamter a_eps
so that it allows to optimise a_eps with kernel

This is nessessary for variational Dirichlet-based GP Classification, as described in the report
'''

import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from check_shapes import inherit_check_shapes

import gpflow
from gpflow.base import InputData
from gpflow.models import GPModel
from gpflow.kullback_leiblers import gauss_kl

from gpflow.config import default_float
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor

# Defines a class DBModel, which inherits from the GPModel class
# representing a Sparse Gaussian Process Regression model with posterior
class DBModel(GPModel, InternalDataTrainingLossMixin):
    ## setup heteroskedastic regression
    ## ================================

    # $\alpha_epsilon$ parameter: 
    # it can be considered as the parameter of a Dirichlet distribution
    # prior to the observation of any label.
    def tilde(self, data, a_eps):
        X, y = data
        a_eps = tf.exp(a_eps)  # ensuring a_eps > 0
        
        # one-hot vector coding
        y_vec = y.astype(int)
        classes = np.max(y_vec).astype(int) + 1
        Y01 = np.zeros((len(y_vec), classes))
        for i in range(len(y_vec)):
            Y01[i, y_vec[i]] = 1
            
        # label transformation
        s2_tilde = tf.math.log(1.0 / (Y01 + a_eps) + 1)
        Y_tilde = tf.math.log(Y01 + a_eps) - 0.5 * s2_tilde
        
        data_tilde = data_input_to_tensor((X, Y_tilde))
        return data_tilde, s2_tilde
    
    def __init__(self, data, lengthscales, variance, a_eps, Z, mean_function=None, **kwargs):
        self.data = data
        
        # parameterise a_eps and Z (wrap)
        self.a_eps = gpflow.Parameter(a_eps, trainable=True)
        self.Z = gpflow.Parameter(Z, trainable=False)
        
        ## GP setup
        ## ====================================
        # set parameters for kernel
        self.len0 = lengthscales
        self.var0 = variance
        kernel = gpflow.kernels.RBF(
            lengthscales=self.len0, variance=self.var0
        )
        
        # set multiclass likelihood
        self.num_latent = num_latent = np.max(data[1]) + 1
        likelihood = gpflow.likelihoods.MultiClass(
            num_classes=self.num_latent, invlink=None, **kwargs
        )

        super().__init__(
            kernel,
            likelihood,
            mean_function,
            num_latent_gps=num_latent,
            **kwargs
        )

        self.mean_function = mean_function or gpflow.mean_functions.Zero()

    def posterior(self):
        '''
        Compute the mean and variance of posterior function.
        Follow the GPR posterior.
        '''
        X, Y_tilde = self.data_tilde
        kff = self.kernel(X, full_cov=True)
        kff += tf.linalg.diag(
            1e-4 * tf.ones(X.shape[0], dtype=default_float())
        )  # use 1e-4 to ensure cholesky decomposition
        noise = tf.linalg.diag(tf.transpose(self.s2_tilde))
        K = kff + noise
        
        L = tf.linalg.cholesky(K)
        invL_y = tf.linalg.triangular_solve(
            L, tf.expand_dims(tf.transpose(Y_tilde), -1), lower=True
        )
        invL_kff = tf.linalg.triangular_solve(L, kff, lower=True)
        
        q_mu = tf.squeeze(
            tf.linalg.matmul(tf.transpose(invL_kff, perm=[0, 2, 1]), invL_y)
        )
        q_cov = kff - tf.linalg.matmul(
            tf.transpose(invL_kff, perm=[0, 2, 1]), invL_kff
        )
        
        return q_mu, q_cov

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        """
        Construct a tensorflow function to compute the bound on divergence.
        """
        self.data_tilde, self.s2_tilde = self.tilde(self.data, self.a_eps)
        X, Y = self.data
        q_mu, q_cov = self.posterior()
        q_sqrt = tf.linalg.cholesky(q_cov)
        
        # compute KL divergence
        kff = self.kernel(X, full_cov=True)
        kff += tf.linalg.diag(
            1e-4 * tf.ones(X.shape[0], dtype=default_float())
        )
        KL = gauss_kl(tf.transpose(q_mu), q_sqrt, kff)

        # compute sum of expectations
        f_mean = tf.transpose(q_mu)
        f_var = tf.transpose(tf.linalg.diag_part(q_cov))
        if Y.ndim == 1:
            var_exp = self.likelihood.variational_expectations(
                X, f_mean, f_var, np.expand_dims(Y, -1)
            )
        else:
            var_exp = self.likelihood.variational_expectations(
                X, f_mean, f_var, Y
            )
        expectation = tf.reduce_sum(var_exp)

        return expectation - KL

    @inherit_check_shapes
    def predict_f(
            self,
            Xnew: InputData,
            full_cov: bool = False,
            full_output_cov: bool = False,
    ):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. Follow the GPR prediction.
        """
        self.data_tilde, self.s2_tilde = self.tilde(self.data, self.a_eps)
        X, Y_tilde = self.data_tilde
        
        kxx = self.kernel(X)
        kxx += tf.linalg.diag(
            1e-4 * tf.ones(X.shape[0], dtype=default_float())
        )
        noise = tf.linalg.diag(tf.transpose(self.s2_tilde))
        K = kxx + noise
        kxn = self.kernel(X, Xnew)
        knn = self.kernel(Xnew)

        L = tf.linalg.cholesky(K)
        invL_y = tf.linalg.triangular_solve(
            L, tf.expand_dims(tf.transpose(Y_tilde), -1), lower=True
        )
        invL_knx = tf.linalg.triangular_solve(L, kxn, lower=True)
        
        fmu = tf.transpose(tf.squeeze(
            tf.linalg.matmul(tf.transpose(invL_knx, perm=[0, 2, 1]), invL_y)
        )) 
        cov = knn - tf.linalg.matmul(
            tf.transpose(invL_knx, perm=[0, 2, 1]), invL_knx
        )
        fcov = tf.transpose(tf.linalg.diag_part(cov))
        
        return fmu, fcov

