# Principle Component Analysis

"""
Do PCA on image patches
Input:
    -X: matrix with image patches as columns
    -k: the number of largest magnitude eigenvalues
Output:
    -V: whitening matrix
    -E: principal component transformation (orthogonal)
    -D: variances of the principal components
"""

import numpy as np


def pca(X, k):
    covarianceMatrix = np.matmul(X, (np.divide(np.transpose(X), np.size(X, 1))))
    covarianceMatrix.astype(float)

    # TODO: Could be problems here

    D, E = np.linalg.eig(covarianceMatrix)
    dsqrtinv = np.real(np.abs(D)**(-0.5))
    V = np.matmul(np.diag(dsqrtinv), np.transpose(E))
    return V
