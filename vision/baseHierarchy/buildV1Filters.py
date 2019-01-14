import math
import numpy as np

"""
Builds the dictionary of S1 features
This method returns an array of S1 (Gabor) filters. Each Gabor filter is a square 2D array. The inner loop runs through 
orientations (4 orientations per scale) while the outer loop runs through scales
"""


def buildV1filters():

    print("Building V1 filters")
    # The finalized set of filters
    filters = []

    # i is the receptive field size. We go from size of 7 to 30 in increments of 2
    # This is a changeable parameter, but 7 to 30 is a good number
    for i in range(7,30,2):
        # Filters for this receptive field size
        filtersThisSize = []
        for o in range(0, 4):
            # Four phases: 0, pi/4, pi/2, 3pi/4
            theta = o * math.pi / 4
            # Template grid for all the different angles in the phase
            x, y = np.mgrid[0:i, 0:i] - i / 2
            # Lower sigma = sharper edged filters, higher sigma = blurry edged filters
            sigma = 0.0036 * i * i + 0.35 * i + 0.18
            lmbda = sigma / 0.8
            gamma = 0.3

            # Fill x2 and y2 (our template grid) with appropriate angles for this phase
            x2 = x * np.cos(theta) + y * np.sin(theta)
            y2 = -x * np.sin(theta) + y * np.cos(theta)
            # Create Gaussian filter at the set orientation
            filter = (np.exp(-(x2 * x2 + gamma * gamma * y2 * y2) / (2 * sigma * sigma)) * np.cos(2 * math.pi * x2 / lmbda))

            filter[np.sqrt(x ** 2 + y ** 2) > (i / 2)] = 0.0
            # Normalize
            filter = filter - np.mean(filter)
            filter = filter / np.sqrt(np.sum(filter ** 2))
            filtersThisSize.append(filter.astype('float'))

        filters.append(filtersThisSize)

    return filters
