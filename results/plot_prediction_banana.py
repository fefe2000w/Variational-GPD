# =============================================================================
# This module is to compare decision boundaries for the first split of banana dataset
# =============================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import math
import gpflow
import GPy as gp

import scipy.io as scio
os.chdir('../src')

from dirichlet_model import DBModel
from heteroskedastic import SGPRh
import datasets

# Sat path to save graph
graph_dir = os.path.join('../results', 'graph')
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

fig, axes = plt.subplots(1, 5, figsize=(20, 8)) 

# Classifier names
titles = ['GPD-fixed', 'GPD-variational', 'LA', 'EP', 'VI']

# Extract data
database = scio.loadmat("../datasets/benchmarks.mat")
benchmarks = database["benchmarks"][0]

def extract_data(name, split_index):
    data = database[name][0, 0]
    X, y, Xtest, ytest = datasets.get_split_data(data, name, 0)
    X, Xtest = datasets.normalise_oneminusone(X, Xtest)
    
    return X, y, Xtest, ytest

for benchmark in benchmarks:
    name = benchmark[0]
    if name != 'banana':
        continue
    X, y, Xtest, ytest = extract_data(name, 0)

plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# Make meshgrid
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]

y_vec = y.astype(int)
classes = np.max(y_vec).astype(int) + 1
Y = np.zeros((len(y_vec), classes))
for i in range(len(y_vec)):
    Y[i, y_vec[i]] = 1

# Label transformation
s2_tilde = np.log(1.0 / (Y + 0.1) + 1)
Y_tilde = np.log(Y + 0.1) - 0.5 * s2_tilde

ymean = np.log(Y.mean(0)) + np.mean(Y_tilde - np.log(Y.mean(0)))
dim = X.shape[1]
Y_tilde = Y_tilde - ymean
var0 = np.var(Y_tilde)
len0 = np.mean(np.std(X, 0)) * np.sqrt(dim)

## For GPD-fixed
kernel = gpflow.kernels.RBF(lengthscales=len0, variance=var0)
GPD_fixed = SGPRh((X,Y_tilde), kernel, s2_tilde, X)
opt = gpflow.optimizers.Scipy()
opt.minimize(GPD_fixed.training_loss, GPD_fixed.trainable_variables)

db_fmu, db_fs2 = GPD_fixed.predict_f(grid_points)
db_fmu = db_fmu + ymean

# Compute probabilities by sampling
db_prob = np.zeros(db_fmu.shape)
source = np.random.randn(1000, classes)
for i in range(db_fmu.shape[0]):
    samples = source * np.sqrt(db_fs2[i, :]) + db_fmu[i, :]
    samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)

    db_prob[i, :] = samples.mean(0)

Z = db_prob[:,1]
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8, vmin=0, vmax=1)  
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
axes[1].set_title('GPD-fixed')

# For GPD-variational
GPD_variational = DBModel((X,y), len0, var0, math.log(0.1), X)
opt.minimize(GPD_variational.training_loss, GPD_variational.trainable_variables)

db_fmu, db_fs2 = GPD_variational.predict_f(grid_points)

# Compute probabilities by sampling
db_prob = np.zeros(db_fmu.shape)
source = np.random.randn(1000, classes)
for i in range(db_fmu.shape[0]):
    samples = source * np.sqrt(db_fs2[i, :]) + db_fmu[i, :]
    samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)

    db_prob[i, :] = samples.mean(0)

Z = db_prob[:,1]
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8, vmin=0, vmax=1)
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
axes[0].set_title('GPD-variational')

# For VI
Y = y.reshape(y.size, 1)
default_len = np.mean(np.std(X, 0)) * np.sqrt(dim)
kernel = gpflow.kernels.RBF(lengthscales=default_len, variance=np.var(Y))
lik = gpflow.likelihoods.Bernoulli()

VI = gpflow.models.VGP(
    (X, Y), kernel=kernel, likelihood=lik, num_latent_gps=classes
)
opt.minimize(VI.training_loss, VI.trainable_variables)
vi_prob, _ = VI.predict_y(grid_points)
Z = vi_prob.numpy()[:, 1]

Z = Z.reshape(xx.shape)

axes[4].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8, vmin=0, vmax=1)
axes[4].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
axes[4].set_title('VI')

# For LA
kernel = gp.kern.RBF(input_dim=dim, ARD=False, lengthscale=default_len)
lik = gp.likelihoods.Bernoulli()
laplace = gp.inference.latent_function_inference.Laplace()
LA = gp.core.GP(
    X, Y, kernel=kernel, likelihood=lik, inference_method=laplace
)
LA.optimize()
Z = LA.predict(grid_points)[0]
Z = Z.reshape(xx.shape)

axes[2].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8, vmin=0, vmax=1)
axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
axes[2].set_title('LA')

# For EP
kernel = gp.kern.RBF(input_dim=dim, ARD=False, lengthscale=default_len)
lik = gp.likelihoods.Bernoulli()
laplace = gp.inference.latent_function_inference.EP()
EP = gp.core.GP(
    X, Y, kernel=kernel, likelihood=lik, inference_method=laplace
)
EP.optimize()
Z = EP.predict(grid_points)[0]
Z = Z.reshape(xx.shape)

axes[3].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8, vmin=0, vmax=1)
axes[3].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
axes[3].set_title('EP')

plt.tight_layout()
save_path = os.path.join(graph_dir, "prediction.pdf")
plt.savefig(save_path, format="pdf")

plt.show()

