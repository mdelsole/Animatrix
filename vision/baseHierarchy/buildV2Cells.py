import numpy as np
import math
from util import sumFilter

"""
V2 builds complex cells. These cells pool over inflowing units in V1 from the same orientation selectivity, but at
slightly different positions and scales to increase the cells' tolerance.

V2 performs a max operation, i.e. for each V2 (complex) cell, the cell in its inflow that fires most is chosen

Inputs:
    -img: The input image
    -filters: Matrix of filters created in V1
    -filterSizes: Contains sizes of the filters
    -v2Scale: Defines the scale bounds, i.e. the filter sizes over which a local max is taken to get V2 cell responses
    -v2Space: Receptive field size of V2 cells
    -v2Overlap: Defines the overlap in receptive field between each V2 cell
"""


def buildV2Cells(img, filters, filterSizes, v2Space, v2Scale, v2Overlap, includeBorders):

    # Size of scale bounds
    nBands = np.size(v2Scale)-1
    # Last element in c1Scale is max scale + 1
    nScales = v2Scale[-1]-1
    nFilters = math.floor(np.size(filterSizes)/nScales)

    scalesInThisBand = []
    # Define the scale bounds of each band
    for i in range(nBands):
        scalesInThisBand.append(np.array([v2Scale[i], v2Scale[i + 1] - 1]))

    # Rebuild all filters of all sizes
    nFilts = np.size(filterSizes)
    # Square filter?
    sqFilter = []
    for i in range(nFilts):
        sqFilter.append(np.reshape(filters[1:filterSizes(i)**2, 1], (filterSizes(i), filterSizes(i))))
        # invert
        sqFilter[i] = np.flipud(sqFilter[i])
        sqFilter[i] = np.fliplr(sqFilter[i])

    # Compute all s1 filter responses

    # First calculate normalizations for the usable filter sizes
    imgSquared = img**2
    # Because there are 4 (orientations) of each of the 17 sizes, we just want one of each size
    ufilterSizes = np.unique(filters)
    s1Norm = []
    for i in ufilterSizes:
        s1Norm[i] = sumFilter.sumFilter(imgSquared, (i-1)/2)**2
        # To avoid a divide by zero later
        s1Norm[i][np.where(s1Norm == 0)] = 1

    # Next, apply filters
    iUFilterIndex = 0

