# Preprocess the inputted image

import numpy as np
import cv2
from vision.baseHierarchy import buildV1Cells

filename = "/Users/Michael/Documents/Animatrix/testImages/lena.jpg"

print("Processing image "+filename)
# Load the image in grayscaled
img = cv2.imread(filename, 0)

print("Building V1 Filters")
# 4 orientations for gabor filters
orientations = np.array([90, -45, 0, 45])
# Initialize the different receptive field sizes
rfsizes = np.arange(7, 41, 2)
# Initialize scaling factors, tuning the wavelength of the sinusoidal factor 'lambda' in relation to each of the rfsizes
div = np.arange(4, 3.195, -0.05)
buildV1Cells.buildV1Cells(orientations, rfsizes, div)

print("Building C1")
# Defining 8 scale bands
c1Scale = np.arange(1, 18, 2)
# Defining spatial pooling range for each scale band
c1Space = np.arange(8, 24, 2)

