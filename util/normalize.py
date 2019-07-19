import numpy as np

def normalize(X):
    X = np.subtract(X, np.mean(X,0))
    patchArray = np.divide(X, np.sqrt(np.var(X,0))+10)
    return patchArray