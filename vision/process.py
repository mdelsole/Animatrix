# Preprocess the inputted image

import numpy as np
import cv2
from vision.baseHierarchy.architecture import v1SimpleCells
from vision.baseHierarchy.architecture import v1ComplexCells

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = "/Users/Michael/Documents/Animatrix/testImages/lena.jpg"
path = "/Users/Michael/Documents/Animatrix/vision/baseHierarchy/architecture/connectionArrays/"

print("Processing image "+filename)
# Load the image in grayscaled
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)



# Commented out because it only needs to be run once to form the npy file
"""
# Build V1 Simple Cells
print("\nBuilding V1 Simplex Cells")

# 4 orientations for gabor filters
orientations = np.array([90, -45, 0, 45])
# Initialize the different receptive field sizes
rfsizes = np.arange(7, 41, 2)
# Initialize scaling factors, tuning the wavelength of the sinusoidal factor 'lambda' in relation to each of the rfsizes
div = np.arange(4, 3.195, -0.05)


filters, filterSizes = v1SimpleCells.v1SimpleCells(orientations, rfsizes, div)
# Save, so we don't need to run it every time
np.save(path + 'filters.npy', filters)
np.save(path + 'filterSizes.npy', filterSizes)
"""


# Build V1 Complex Cells
print("\nBuilding V1 Complex Cells")

# Defining 8 scale bands
c1Scale = np.arange(1, 18, 2)
# Defining spatial pooling range (i.e. receptive field) for each scale band
c1Space = np.arange(8, 24, 2)
# Define the sliding speed
c1Overlap = 2

filters = np.load(path + '/filters.npy')
filterSizes = np.load(path + '/filterSizes.npy')


v1ComplexCells.v1ComplexCells(img, filters, filterSizes, c1Space, c1Scale, c1Overlap)