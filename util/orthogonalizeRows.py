# Orthogonalize the rows of a matrix

import numpy as np
from scipy.linalg import fractional_matrix_power as frac

def orthogonalizeRows(W):

    wOrthogonalized = np.matmul(np.real(frac(np.matmul(W, np.transpose(W)),-0.5)),W)

    return wOrthogonalized