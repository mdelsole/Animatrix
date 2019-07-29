import numpy as np
from util import im2col, normalize


def v1SimpleCell(X, pmat, bases, patchsz):

    # Concatenate the features
    row = np.shape(X)[0]
    col = np.shape(X)[1]
    numBases = np.shape(X)[2]

    descrs = np.zeros((np.linalg.matrix_power(patchsz,2) * numBases, (row - patchsz) * (col - patchsz)), 'float')
    for i in range(numBases):
        descrs[i*(patchsz**2): (i+1)*patchsz**2] = im2col.im2col_sliding(X[:,:,i], patchsz)

    # Normalize the features
    descrs = normalize.normalize(descrs)

    # Project to the principal components space
    descrs = np.matmul(pmat, descrs)

    # Calculate S1 map
    S = descrs*bases

    # Convert back to 3D arrays
    S = np.reshape(S, (row-patchsz, col-patchsz, np.size(bases, 1)))

    return S
