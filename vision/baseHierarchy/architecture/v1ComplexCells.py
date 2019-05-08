import numpy as np
import math
from util import sumFilter
import matplotlib.pyplot as plt
from matplotlib import gridspec



"""
V1's complex cells. These cells pool over inflowing simple cells from the same orientation selectivity, but at
slightly different positions and scales to increase the cells' tolerance.

Complex cells perform a max operation, i.e. for each complex cell the cell in its inflow that fires most is chosen.
In other words, the response y of a complex unit corresponds to the response of the strongest of its afferents.


Inputs:
    -img: The input image
    -filters: Matrix of filters created in V1 simple cells
    -filterSizes: Matrix containing sizes of the filters
    
Pooling over space and scale are done in 1 operation together
    -cellScale: Defines the scale bounds, i.e. the filter sizes over which a local max is taken to get complex cell 
     responses. 
    -receptiveFieldSize: Receptive field size of the complex cells
    -v2Overlap: Defines the overlap in receptive field between each V2 cell
    
Returns:
    -c1: a cell array [1 nBands], contains the C1 responses for img
    -s1: a cell array [1 nBands], contains the S1 responses for img
"""


def v1ComplexCells(img, filters, filterSizes, receptiveFieldSize, cellScaleRange, v2Overlap, includeBorders):

    # Size of scale bands, the filter sizes over which a max is taken to get the complex cell response
    nBands = np.size(cellScaleRange)
    # Last element in c1Scale is max scale
    nScales = cellScaleRange[-1]
    # Number of filters complex cells max over = #filterSizes/nScales
    nFilters = int(math.floor(np.size(filterSizes)/nScales))

    # Visualize each filter
    spec = gridspec.GridSpec(nrows=5, ncols=39)
    fig = plt.figure()

    scalesInThisBand = []
    # Define the scale bounds of each band
    for i in range(nBands):
        scalesInThisBand.append(np.array([cellScaleRange[i], cellScaleRange[i + 1]]))

    # Rebuild all the filters of all sizes
    nFilts = np.size(filterSizes)
    # Square filter?
    sqFilter = []
    for i in range(nFilts):
        sqFilter.append(np.reshape(filters[1:filterSizes(i)**2, 1], (filterSizes(i), filterSizes(i))))
        # invert
        sqFilter[i] = np.flipud(sqFilter[i])
        sqFilter[i] = np.fliplr(sqFilter[i])


    """

    # Compute all filter responses (v1 simple cells)

    # First calculate normalizations for the usable filter sizes
    imgSquared = img**2
    # Because there are 4 (orientations) of each of the 17 sizes, we just want one of each size
    ufilterSizes = np.unique(filters)
    s1Norm = []
    for i in ufilterSizes:
        s1Norm[i] = sumFilter.sumFilter(imgSquared, (i-1)/2)**2
        # To avoid a divide by zero later
        s1Norm[i][np.where(s1Norm == 0)] = 1

    # Then, apply filters
    iUFilterIndex = 0
    s1 = []
    for iBands in range(nBands):
        for iScale in range(len(scalesInThisBand)):
            for iFilt in range(nFilters):
                iUFilterIndex += 1
                #Finish Later

    # Calculate local pooling (v1 complex cells)

    # First, pool over scales within band
    # For each band (range)
    for iBand in range(nBands):
        # For each filt in each band
        for iFilt in range(nFilters):
            # Finish later
            i = 1

    # Then, pool over local neighborhood
    for iBand in range(nBands):
        # Fix poolRange
        poolRange = (receptiveFieldSize(iBand))
        for iFilt in range(nFilters):
            # Finish later
            i = 1
            
            """
