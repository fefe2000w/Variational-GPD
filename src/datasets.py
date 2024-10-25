'''
This module assists data normalisation,
referring to https://github.com/dmilios/dirichletGPC/blob/master/src/datasets.py
'''

import numpy as np

def normalise_oneminusone (X, Xtest):
    minx = np.min(X, 0)
    maxx = np.max(X, 0)
    ranges = maxx - minx
    ranges[ranges == 0] = 1 # to avoid NaN
    
    X = (X - minx) / ranges
    Xtest = (Xtest - minx) / ranges
    X = X * 2 - 1
    Xtest = Xtest * 2 - 1
    
    return X, Xtest

def get_split_data(data, name, split_index):
    x = data["x"]
    t = data["t"]
    print(name, split_index)  # e.g. banana 0 means the first split of benchmark banana
    
    train_idx = data["train"][split_index] - 1
    test_idx = data["test"][split_index] - 1
    
    X = x[train_idx.flatten(), :]
    y = t[train_idx.flatten()]
    Xtest = x[test_idx.flatten(), :]
    ytest = t[test_idx.flatten()]
    y = np.where(y == -1, 0, y)
    ytest = np.where(ytest == -1, 0, ytest)
    
    return X, y, Xtest, ytest
