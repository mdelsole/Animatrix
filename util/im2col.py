import numpy as np


def im2col_sliding(A, BSZ):

    m, n = A.shape
    result = np.empty((0, BSZ[1]*BSZ[0]))

    for j in range(m-BSZ[1]+1):
        for i in range(n-BSZ[0]+1):
            # Cols = [j to blockWidth+j]
            cols = np.arange(j, BSZ[1]+j)
            z = np.take(A, cols, axis=1)
            # Cut off the end, then ravel
            newCol = np.ravel(z[i:BSZ[0]+i, :], order='F')
            B = np.reshape(newCol, (1, -1))
            result = np.concatenate((result, B), axis=0)

    return result.T


def im2col_distinct(A, BSZ):
    m, n = A.shape
    result = np.empty((0, BSZ[1] * BSZ[0]))
    j = 0
    # Determine padding
    padRows = m % BSZ[0]
    padCols = n % BSZ[1]
    if padRows != 0:
        padRows = BSZ[0] - padRows
    if padCols != 0:
        padCols = BSZ[1] - padCols
    A = np.pad(A, [(0, padRows),(0, padCols)], 'constant')

    while j < m:
        i = 0
        while i < n:
            # Cols = [j to blockWidth+j]
            cols = np.arange(j, BSZ[1] + j)
            z = np.take(A, cols, axis=1, mode='clip')
            # Cut off the end, then ravel
            newCol = np.ravel(z[i:BSZ[0] + i, :], order='F')
            B = np.reshape(newCol, (1, -1))
            result = np.concatenate((result, B), axis=0)
            i += BSZ[0]
        j += BSZ[1]

    return result.T
