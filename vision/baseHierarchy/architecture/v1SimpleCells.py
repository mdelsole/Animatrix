import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# View whole array, for debugging purposes
np.set_printoptions(threshold=np.nan)

"""
Builds the dictionary of V1 simple cells
This method returns an array of simple Gabor filters. Each Gabor filter is a square 2D array. The inner loop runs through 
orientations (4 orientations per scale) while the outer loop runs through scales
"""


def v1SimpleCells(orientations, rfsizes, div):

    numOrientations = np.size(orientations)
    numRfsizes = np.size(rfsizes)
    numFilters = numRfsizes*numOrientations

    # Arrays: [how many down][how many across]

    # Spatial aspect ratio
    gamma = 0.3
    # Array of the filter sizes (17 different sets of 4 orientations, 1 for each rfsize)
    filterSizes = np.zeros((numFilters,1))
    # Storage place for the finalized filters
    # Column = biggest rfsize, squared because rfs are square. Row = for each different rfsize
    filters = np.zeros((np.max(rfsizes)**2,numFilters))

    for k in range(numRfsizes):
        for o in range(numOrientations):
            # Angle
            theta = orientations[o]*math.pi/180
            # Size of current filter
            filterSize = rfsizes[k]
            # Center of filter, i.e. midpoint
            center = int(math.ceil(filterSize/2))
            # Size on the left and right. Used for location. -1 b/c coordinates start at 0
            filterSizeL = center
            filterSizeR = filterSize-filterSizeL
            # Lambda = wavelength
            lmbda = (filterSize*2)/div[k]
            # Lower sigma = sharper edged filters, higher sigma = blurry edged filters
            sigma = (lmbda)*0.8
            sigmaSquared = (sigma)**2

            f = np.zeros(((filterSizeR-(-filterSizeL)), (filterSizeR-(-filterSizeL))))
            #print("F Shape: ", f.shape)

            # Apply filter over the receptive field. Rfs are square, thus use same range().
            for i in range(-filterSizeL, filterSizeR):
                for j in range(-filterSizeL, filterSizeR):
                    # If out of bounds, filter is zero
                    if math.sqrt(i**2+j**2) > filterSize/2:
                        e = 0
                    else:
                        # X coordinate for the filter
                        x = i*math.cos(theta) - j*math.sin(theta)
                        # Y coordinate for the filter
                        y = i*math.sin(theta) + j*math.cos(theta)
                        # Filter at that x,y coordinate
                        e = math.exp(-((x**2)+((gamma**2)*(y**2)))/(2*sigmaSquared))*math.cos(2*math.pi*x/lmbda)
                    #print("J, I: ", j+center, i+center)
                    f[j+center][i+center] = e
            # Normalize
            f = f - np.mean(f)
            f = f/np.sqrt(np.sum(f**2))
            #print(filterSizes.shape)

            # Ith filter
            iFilter = numOrientations*(k) + o
            filters[0:filterSize**2, iFilter] = np.reshape(f, (filterSize**2))
            filterSizes[iFilter] = filterSize

    np.set_printoptions(precision=4)
    numCells = filters.shape[0]
    print(numCells)
    SIDE = np.ceil(np.sqrt(numCells)).astype(int)
    RFSIZE = int(np.sqrt(filters.shape[1]))

    ax = plt.subplot(SIDE, SIDE, 505 + 1)
    ax.matshow(np.reshape(filters[505, :], (RFSIZE, RFSIZE)), cmap='Greys_r')
    ax.set_axis_off()

    plt.draw()
    #imgplot = plt.imshow(filterSizes, cmap=plt.get_cmap('gray'))
    plt.savefig('RFs.png', bbox_inches='tight')