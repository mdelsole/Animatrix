# Principle Component Analysis

import numpy as np

def pca(X, k):
    covarianceMatrix = float(X * (np.divide(np.transpose(X), np.size(X,1))))
    E, D = np.linalg.eigh(covarianceMatrix, k)

    d = np.diag(D)
    dsqrtinv = np.real(d**(-0.5))
    V = np.diag(dsqrtinv) * np.transpose(E)
    return V
