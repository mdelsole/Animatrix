import numpy as np
import math

"""
V2 builds complex cells. These cells pool over inflowing units in V1 from the same orientation selectivity, but at
slightly different positions and scales to increase the cells' tolerance.

V2 performs a max operation, i.e. for each V2 (complex) cell, the cell in its inflow that fires most is chosen

Inputs:
    -img: The input image
    -filters: matrix of filters created in V1
    -filterSizes: contains sizes of the filters
    -c1Scale: defines the scale bands, a group of filter sizes over which a local max is taken to get C1 unit responses
"""


def buildV2Cells(img, filters, filterSizes, c1Space, c1Scale, c10L, includeBorders):
    nbands = np.size(c1Scale)-1
    # Last element in c1Scale is max scale + 1
    nScales = c1Scale[-1] - 1
    nFilters = math.floor(np.size(filterSizes)/nScales)
