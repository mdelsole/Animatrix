# Display network A in m rows and n columns
# A = basis function matrix

import numpy as np
import matplotlib.pyplot as plt


def visualize(A, m, n):

    plt.ion()
    np.set_printoptions(precision=4, suppress=True)

    NBCELLS = 36
    SIDE = np.ceil(np.sqrt(NBCELLS)).astype(int)
    RFSIZE = 10

    fig = plt.figure()

    for nn in range(NBCELLS):
        ax = plt.subplot(SIDE, SIDE, nn + 1)
        ax.matshow(np.reshape(A[:, nn], (RFSIZE, RFSIZE)), cmap='Greys_r')
        ax.set_axis_off()

    plt.draw()
    plt.savefig('RFs.png', bbox_inches='tight')