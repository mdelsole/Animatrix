import numpy as np


def pool(map, rowRatio, colRatio, rowRest, colRest, poolMethod):
    row, col, numBases = np.size(map)

    # Trim the borders

    # TODO: Figure this out
    # map = np.delete(map, row-(rowRest-round(rowRest/2)), 1)
    # map = map[row-(rowRest-round(rowRest/2)):, ]

    # Evaluate the size again after adjusting borders
    row, col, numBases = np.size(map)

    # Check if the size of map is divisible by the block
    if row%rowRatio or col%colRatio:
        print("Trimming is incorrect")

    # Use im2col
