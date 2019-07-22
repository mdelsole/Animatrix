"""
Inputs:
    -Z: Whitened image patch data
    -n = 1:windowsize**2-1 : number of independent components to be estimated
"""
import numpy as np
from util import orthogonalizeRows
# np.set_printoptions(threshold=np.nan)



def ica(Z, n):

    ######## Parameters ########
    convergenceCriterion = 1e-4
    # Take to 2000
    maxIters = 200


    ######## Initialize ########
    # Create random initial value of W, and orthogonalize it
    W = orthogonalizeRows.orthogonalizeRows(np.random.randn(n, np.size(Z, 0)))
    # Read the sampleSize from data matrix
    N = np.size(Z,1)

    ######## Start Algorithm ########
    print("Doing FastICA")
    iter = 0
    notConverged = True

    while (notConverged and (iter<maxIters)):
        iter += 1
        print("Iteration: ", iter)
        # Store old value
        wOld = W

        # FastICA step

        # Compute estimates of independent components
        Y = np.matmul(W, Z)
        # Use tanh non-linearity
        gY = np.tanh(Y)
        #print(np.matmul(gY, (np.transpose(Z) / N)))

        # The fixed-point step
        # Note that 1-(tanh y)^2 is the derivative of the function tanh y
        W = np.matmul(gY, (np.transpose(Z)/N)) - (np.matmul(np.transpose(np.mean(1-(np.transpose(gY)**2), 0, keepdims=True)),
                                                                      np.ones((1, np.size(W,1))))*W)
        # Orthogonalize rows or decorrelate estimated components
        W = orthogonalizeRows.orthogonalizeRows(W)
        # Check if converged by comparing matrix change with small number, which is scaled with dimensions of the data
        if np.linalg.norm(np.abs(np.matmul(W, np.transpose(wOld)))-np.eye(n), 'fro') < convergenceCriterion*n:
            notConverged = False

    return W

