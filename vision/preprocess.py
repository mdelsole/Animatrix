# Preprocess the inputted image

import numpy as np
import cv2

filename = "/Users/Michael/Documents/Animatrix/testImages/lena.jpg"

print("Processing image "+filename)
# Load the image in grayscaled
img = cv2.imread(filename, 0)

print("Building V1 Filters")
# 4 orientations for gabor filters
orientations = np.array([90, -45, 0, 45])
# Initialize the different receptive field sizes
rfsizes = np.arange(7, 39, 2)


