import math
import numpy as np

"""
Builds the dictionary of S1 features
This method returns an array of S1 (Gabor) filters. Each Gabor filter is a square 2D array. The inner loop runs through 
orientations (4 orientations per scale) while the outer loop runs through scales
"""


def buildV1filters(orientations, rfsizes, div):

    numOrientations = np.size(orientations)
    numRfsizes = np.size(rfsizes)

    # Spatial aspect ratio
    gamma = 0.3
    # Array of the filter sizes
    filterSizes = np.zeros((numRfsizes,1))
    # Storage place for the finalized filters
    # Column = biggest rfsize, squared because rfs are square. Row = for each different rfsize
    filters = np.zeros((np.max(rfsizes)**2,numRfsizes))

    for k in range(numRfsizes):
        for o in range(numOrientations):
            # Angle
            theta = orientations[o]*math.pi/180
            # Size of current filter
            filterSize = rfsizes[k]
            # Center of filter, i.e. midpoint
            center = int(math.ceil(filterSize/2))
            # Size on the left and right. Used for location. -1 b/c coordinates start at 0
            filterSizeL = center-1
            filterSizeR = filterSize-filterSizeL-1
            # Lambda = wavelength
            lmbda = (filterSize*2)/div[k]
            # Lower sigma = sharper edged filters, higher sigma = blurry edged filters
            sigma = (lmbda)*0.8
            sigmaSquared = (sigma)**2

            f = np.zeros(((filterSizeR-(-filterSizeL)+center), (filterSizeR-(-filterSizeL)+center)))

            # Apply filter over the receptive field. Rfs are square, thus use same range()
            for i in range(-filterSizeL, filterSizeR):
                for j in range(-filterSizeL, filterSizeR):
                    # If out of bounds, filter is zero
                    if math.sqrt(i**2+j**2) > rfsizes[k]/2:
                        e = 0
                    else:
                        # X coordinate for the filter
                        x = i*math.cos(theta) - j*math.sin(theta)
                        # Y coordinate for the filter
                        y = i*math.sin(theta) + j*math.cos(theta)
                        # Filter at that x,y coordinate
                        e = math.exp(-((x**2)+(gamma**2)*(y**2))/(2*sigmaSquared))*math.cos(2*math.pi*x/lmbda)
                    f[i+center][j+center] = e
            # Normalize
            f = f - np.mean(f)
            f = f/np.sqrt(np.sum(f**2))

            # Ith filter
            iFilter =  numOrientations*(k-1) + o
            filters[1:(filterSize**2)][iFilter] = np.reshape(f, (filterSize**2, 1))
            filterSizes[iFilter] = filterSize





