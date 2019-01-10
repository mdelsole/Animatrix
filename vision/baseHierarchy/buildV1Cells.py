import math
import numpy as np

def buildV1filters():
    """
    Builds the dictionary of S1 features
    This method returns an array of S1 (Gabor) filters. Each Gabor filter is a square 2D array. The inner loop runs
    through orientations (4 orientations per scale) while the outer loop runs through scales
    """

    print ("Building V1 filters")
    filters = []
    for RFSIZE in range(7,30,2):
        filtersthissize = []
        for o in range(0, 4):
            theta = o * math.pi / 4
            # print "RF SIZE:", RFSIZE, "orientation: ", theta / math.pi, "* pi"
            x, y = np.mgrid[0:RFSIZE, 0:RFSIZE] - RFSIZE / 2
            sigma = 0.0036 * RFSIZE * RFSIZE + 0.35 * RFSIZE + 0.18
            lmbda = sigma / 0.8
            gamma = 0.3
            x2 = x * np.cos(theta) + y * np.sin(theta)
            y2 = -x * np.sin(theta) + y * np.cos(theta)
            myfilt = (np.exp(-(x2 * x2 + gamma * gamma * y2 * y2) / (2 * sigma * sigma))
                      * np.cos(2 * math.pi * x2 / lmbda))
            # print type(myfilt[0,0])
            myfilt[np.sqrt(x ** 2 + y ** 2) > (RFSIZE / 2)] = 0.0
            # Normalized like in Minjoon Kouh's code
            myfilt = myfilt - np.mean(myfilt)
            myfilt = myfilt / np.sqrt(np.sum(myfilt ** 2))
            filtersthissize.append(myfilt.astype('float'))
        filters.append(filtersthissize)
    return filters