import numpy as np
import math
from util import sumFilter
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import convolve2d



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
    -cellScale: Defines the scale bands, i.e. the filter sizes over which a local max is taken to get complex cell 
     responses. Each band contains 2 adjacent filter sizes. Ex: Band 1 contains 7x7 and 9x9
        -The scale band index also determines the size of the simple cell neighborhood (#simplecells x #simplecells) 
        over which the complex cells pool
    -receptiveFieldSize: Receptive field size of the complex cells
    -v2Overlap: Defines the overlap in receptive field between each V2 cell
    
Returns:
    -c1: a cell array [1 nBands], contains the C1 responses for img
    -s1: a cell array [1 nBands], contains the S1 responses for img
"""


def v1ComplexCells(img, filters, filterSizes, receptiveFieldSize, cellScaleRange, v2Overlap):

    # Size of scale bands, the filter sizes over which a max is taken to get the complex cell response
    nBands = np.size(cellScaleRange)-1
    # print(nBands)
    # Last element in c1Scale is max scale
    nScales = cellScaleRange[-1]-1
    # print(nScales)
    # Number of filters complex cells max over = #filterSizes/nScales
    nFilters = int(math.floor(np.size(filterSizes)/nScales))
    # print(nFilters)

    scalesInThisBand = []
    # Define the scale bounds of each band
    for i in range(nBands):
        scalesInThisBand.append(np.array([cellScaleRange[i], cellScaleRange[i+1]-1]))

    # Rebuild all the filters of all sizes
    nFilts = np.size(filterSizes)
    # Our unpacked square filters
    sqFilter = []
    for i in range(nFilts):
        intFiltSize = int(filterSizes[i])
        sqFilter.append(np.reshape(filters[:(intFiltSize**2), i], (intFiltSize, intFiltSize)))

        # invert
        sqFilter[i] = np.flipud(sqFilter[i])
        sqFilter[i] = np.fliplr(sqFilter[i])


    ##### Compute all filter responses (v1 simple cells) #####


    """"# First calculate normalizations for the usable filter sizes
    imgSquared = img**2
    # Because there are 4 (orientations) of each of the 17 sizes, we just want one of each size
    ufilterSizes = np.unique(filters)
    print(np.size(ufilterSizes))
    s1Norm = []
    for i in range(np.size(ufilterSizes)):
        s1Norm.append(sumFilter.sumFilter(imgSquared, int((i-1)/2))**(1/2))
        print(i)
        # To avoid a divide by zero later
        s1Norm[i][np.where(s1Norm == 0)] = 1
    """

    # Then, apply filters
    iUFilterIndex = 0
    print(np.size(sqFilter))
    s1 = [[[]]]
    for iBands in range(nBands):
        for iScale in range(len(scalesInThisBand)):
            for iFilt in range(nFilters):
                print("ibands: ", iBands, ", iScale: ", iScale, ", iFilt: ", iFilt)
                s1.append(abs(convolve2d(img,sqFilter[iUFilterIndex],mode='same')))
                # TODO: Remove borders?
                # Normalize
                #s1[iBands][iScale][iFilt] /= s1Norm[filterSizes[iUFilterIndex]]
                iUFilterIndex += 1


    ##### Calculate local pooling (v1 complex cells) #####


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
            

