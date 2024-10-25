# =============================================================================
# This example code refers to Milios et al. (2018) at:
# https://github.com/dmilios/dirichletGPC
# The code file is used to check if variational GPD returns similar prediction 
# to the authors' original model
# =============================================================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gpflow
import numpy as np
import matplotlib.pylab as plt
import math
import sys

sys.path.insert(0, "src")
import dirichlet_model


N = 20  # training data
np.random.seed(1235)


## create synthetic dataset
## ===============================
xmax = 15
X = np.random.rand(N, 1) * xmax
Xtest = np.linspace(0, xmax * 1.5, 200).reshape(-1, 1)
Z = X.copy()

y = np.cos(X.flatten()) / 2 + 0.5
y = np.random.rand(y.size) > y
y = y.astype(int)
if np.sum(y == 1) == 0:
    y[0] = 1
elif np.sum(y == 0) == 0:
    y[0] = 0
    

## GP setup and hyperparam optimisation
## ====================================
len0 = np.std(X, axis=0)
var0 = 1.0
model = dirichlet_model.DBModel(data=(X, y), lengthscales=len0, variance=var0, a_eps=math.log(0.1), Z=Z)

opt = gpflow.optimizers.Scipy()
print("\nloglik (before) =", model.maximum_log_likelihood_objective())

print(model.trainable_parameters)
opt.minimize(model.training_loss, model.trainable_variables)
print("loglik  (after) =", model.maximum_log_likelihood_objective())
print(model.trainable_parameters)

## GP prediction
## =============
fmu, fs2 = model.predict_f(Xtest)

lb = fmu - 2 * np.sqrt(fs2)
ub = fmu + 2 * np.sqrt(fs2)

# Estimate mean and quantiles of the Dirichlet distribution through sampling
q = 95
mu_dir = np.zeros([fmu.shape[0], 2])
lb_dir = np.zeros([fmu.shape[0], 2])
ub_dir = np.zeros([fmu.shape[0], 2])
source = np.random.randn(1000, 2)
for i in range(fmu.shape[0]):
    samples = source * np.sqrt(fs2[i, :]) + fmu[i, :]
    samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
    Q = np.percentile(samples, [100 - q, q], axis=0)
    mu_dir[i, :] = samples.mean(0)
    lb_dir[i, :] = Q[0, :]
    ub_dir[i, :] = Q[1, :]


## Plotting results
## ================

plt.figure(figsize=(12, 4))

# to plot the tranformed labels and their standard deviation

plt.subplot(2, 2, 1)
plt.plot(Xtest, fmu[:, 0], "b", label="Posterior")
plt.fill_between(Xtest.flatten(), ub[:, 0], lb[:, 0], facecolor="0.75")
plt.title("Class 0")
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(Xtest, mu_dir[:, 0], "b", label="Prediction")
plt.plot(X, 1 - y, "o", label="Data")
plt.fill_between(Xtest.flatten(), ub_dir[:, 0], lb_dir[:, 0], facecolor="0.75")
plt.title("Class 0")
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(Xtest, fmu[:, 1], "b", label="Posterior")
plt.fill_between(Xtest.flatten(), ub[:, 1], lb[:, 1], facecolor="0.75")
plt.title("Class 1")
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(Xtest, mu_dir[:, 1], "b", label="Prediction")
plt.plot(X, y, "o", label="Data")
plt.fill_between(Xtest.flatten(), ub_dir[:, 1], lb_dir[:, 1], facecolor="0.75")
plt.title("Class 1")
plt.xticks([], [])
plt.legend()

plt.show()


