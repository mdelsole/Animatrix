import numpy as np
import os
import cv2
from vision.baseHierarchy import v1SimpleCell
from vision.algorithms import pooling


"""
V1's complex cells. These cells pool over inflowing simple cells from the same orientation selectivity, but at
slightly different positions and scales to increase the cells' tolerance.

Complex cells perform a max operation, i.e. for each complex cell the cell in its inflow that fires most is chosen.
In other words, the response y of a complex unit corresponds to the response of the strongest of its afferents.

Inputs:
    -
"""


def v1ComplexCell(directory, patchDirectory, bases, pmat, ratio, minImgSize):

    # TODO: Enable Parallel Computing

    # List all image files in the folder
    fileList = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    r = np.random.permutation(len(fileList))

    for i in range(min(100,len(r))):
        img = cv2.imread(os.path.join(directory, fileList[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, minImgSize)

        # Calculate S1 map
        S1map = v1SimpleCell.v1SimpleCell(img, pmat, bases, np.sqrt(np.size(pmat, 1)))

        # Calculate C1 map
        C1map = pooling.pool(S1map, ratio, ratio, np.size(S1map, 0)%ratio, np.size(S1map, 1)%ratio)

    np.save('C1.npy', C1map)

