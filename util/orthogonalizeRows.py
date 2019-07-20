# Orthogonalize the rows of a matrix

import numpy as np

def orthogonalizeRows(W):

    wOrthogonalized = np.matmul((np.abs(np.matmul(W, np.transpose(W)))**(-0.5)),W)
    # wOrthogonalized = np.matmul((np.matmul(W, np.transpose(W))**(-0.5)),W)
    # wOrthogonalized[np.isnan(wOrthogonalized)] = 0

    return wOrthogonalized