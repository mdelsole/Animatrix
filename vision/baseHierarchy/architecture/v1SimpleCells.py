import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# View whole array, for debugging purposes
from matplotlib import gridspec

np.set_printoptions(threshold=np.nan)

"""
Builds the dictionary of V1 simple cells
This method returns an array of simple Gabor filters. Each Gabor filter is a square 2D array. The inner loop runs through 
orientations (4 orientations per scale) while the outer loop runs through scales
"""


def v1SimpleCells(orientations, rfsizes, div):

    numOrientations = np.size(orientations)
    print("numOrientations: ", numOrientations)
    numRfsizes = np.size(rfsizes)
    print("numRFsizes: ", numRfsizes)
    #Times 2 because 2 phases
    numFilters = numRfsizes*numOrientations*2
    print("numFilters: ", numFilters)

    spec = gridspec.GridSpec(nrows=5, ncols=39)
    fig = plt.figure()

    # Arrays: [how many down][how many across]

    # Spatial aspect ratio. Controls the height of the function
    gamma = 0.3
    # Array of the filter sizes (17 different sets of 4 orientations, 1 for each rfsize)
    filterSizes = np.zeros((numFilters,1))
    print(filterSizes.shape)
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
            #print("Center: ", center)
            # Size on the left and right. Used for location. -1 b/c coordinates start at 0
            filterSizeL = center-1
            #print("filterSizeL: ", filterSizeL)
            filterSizeR = filterSize-filterSizeL-1
            #print("filterSizeR: ", filterSizeL)
            # Lambda = wavelength. Larger wavelengeth = thicker stripes. We make lambda proportional to the filter size
            # because we want thicker stripes when we increase the size of the filter
            lmbda = (filterSize*2)/div[k]
            #print("Lambda: ", lmbda)
            # Lower sigma = sharper edged filters, higher sigma = blurry edged filters
            sigma = (lmbda)*0.8
            sigmaSquared = (sigma)**2

            f = np.zeros(((filterSizeR-(-filterSizeL)+1), (filterSizeR-(-filterSizeL)+1)))
            #print("F Shape: ", f.shape)

            # Form the filter (draw it) over the receptive field. Rfs are square, thus use same range().
            for i in range(-filterSizeL, filterSizeR+1):
                for j in range(-filterSizeL, filterSizeR+1):
                    # If out of bounds, filter is zero
                    if math.sqrt(i**2+j**2) > filterSize/2:
                        #print("Got Here, I, j, Fs/2: ", math.sqrt(i**2+j**2), filterSize/2)
                        e = 0
                    else:
                        # X coordinate for the filter
                        x = i*math.cos(theta) - j*math.sin(theta)
                        # Y coordinate for the filter
                        y = i*math.sin(theta) + j*math.cos(theta)
                        # Filter at that x,y coordinate. This is the equation for a gabor filter
                        e = math.exp(-((x**2)+((gamma**2)*(y**2)))/(2*sigmaSquared))*math.cos(2*math.pi*x/lmbda)
                    #print("J, I: ", j+center, i+center)
                    f[j+center-1,i+center-1] = e
            # Normalize
            f = f - np.mean(f)
            f = f/np.sqrt(np.sum(f**2))
            #print(filterSizes.shape)

            # Ith filter
            iFilter = numOrientations*(k) + o
            #print("FilterSize: ", filterSize)
            #print(filterSizes)
            #F was a 1D
            filters[0:filterSize**2, iFilter] = np.reshape(f, (filterSize**2))
            filterSizes[iFilter] = filterSize

            #Visualize it
            ax = fig.add_subplot(spec[o,k])
            ax.matshow(np.reshape(filters[:(filterSize**2), iFilter], (filterSize, filterSize)), cmap='Greys_r')
            ax.set_axis_off()

    #print(filters[:,0])
    print("Filters: ", filters.shape)
    print("FilterSizes: ", filterSizes.shape)
    numCells = filters.shape[1]
    print("NumFilters: ", numCells)
    RFSIZE = int(np.sqrt(filters.shape[0]))
    print(RFSIZE)

    plt.draw()
    #imgplot = plt.imshow(filterSizes, cmap=plt.get_cmap('gray'))
    plt.savefig('RFs.png', bbox_inches='tight')
