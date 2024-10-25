# Variational-GPD
This repository is for individual report Variational Dirichlet-based Classification, which is submitted for the course COMP8755

## Prerequisites
GPflow latest version, GPy latest version

## Main scripts
### 1. Simple example of variational Dirichlet-based Classification
**dirichletGP_example.py**
is a demonstration of variational Dirichlet-based classification, where a synthetic dataset is used.

### 2. Evaluation experiments
**evaluation_experiment.py** 
performs evaluation of the following methods:
(applying GPflow)
  - GPD_variational: the variational Dirichlet-based GPC with a_eps embedded
  - GPD_fixed: the original Dirichlet-based GPC with fixed a_eps=0.1
  - GPD_searched: the original Dirichlet-based GPC with a_eps optimised by random search
  - VI: the GPC approximated using variational inference (VGP)
(applying GPy)
  - LA: the GPC approximated using laplace approximation
  - EP: the GPC approximated using expectation propagation

Various metrics (error rate, MNLL, ECE) are recorded and the total time required for training and predicting.

The results can be visualised by running the **plot_comparison_result.py** 
script in the _results_ directory.

The experiments are based on 10 splits of each benchmark dataset which saved in _datasets_ directory


## Source code
The **src** directory contains the following modules:
  - _dirichlet_model.py_:
  Implementation of variational GPD so that it incorporates a_eps as trainable parameter

  - _heteroskedastic.py_:
  Replicated version of Milios et al. (2018)
  Implementation of heteroskedastic GP regression so that it admits a different vector of noise values for each output dimension.

  - _evaluation.py_: code for the evaluation experiments.
  Used by **evaluation_experiment.py**

  - _datasets.py_: helper functions to normalise data
  Used by **evaluation_experiment.py**

## Results
The **results** directory contains the following:
  - **plot_comparison_result.py**: 
  Script that plots the comparison of Time, Error rate, MNLL, and ECE for all algorithms.

  - **plot_moment _matching.py**: 
  script that plots log-pdf of Lognormal and Gamma distributions with same alphas.

  - **plot_prediction_banana.py**
  script that plots decision boundaries for the first split of banana dataset using GPD-variational, GPD-fixed, VI, LA, EP


## Available datasets
The **datasets** directory contains 13 benchmarks derived from the UCI, DELVE and STATLOG repositories
More details are in **README.md** in _datasets_ directory.
The experiments are based on 11 datasets which are binary, excepting 'image' and 'splice'

## References
dmilios (2018). GitHub - dmilios/dirichletGPC. [online] GitHub. Available at: https://github.com/dmilios/dirichletGPC
GitHub. (2015). tdiethe/gunnar_raetsch_benchmark_datasets: Gunnar Raetsch’s Benchmark Datasets. [online] Available at: https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets
GPflow (2016). GPflow/gpflow at v2.9.1 · GPflow/GPflow. [online] GitHub. Available at: https://github.com/GPflow/GPflow/tree/v2.9.1/gpflow
SheffieldML (2024). GPy/GPy at devel · SheffieldML/GPy. [online] GitHub. Available at: https://github.com/SheffieldML/GPy/tree/devel/GPy
