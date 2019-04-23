# Preprocess the inputted image

import numpy as np
import cv2
from vision.baseHierarchy.architecture import v1SimpleCells
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = "/Users/Michael/Documents/Animatrix/testImages/lena.jpg"

print("Processing image "+filename)
# Load the image in grayscaled
img = cv2.imread(filename)



print("Building V1 Filters")
# 4 orientations for gabor filters
orientations = np.array([90, -45, 0, 45])
# Initialize the different receptive field sizes
rfsizes = np.arange(7, 39, 2)
# Initialize scaling factors, tuning the wavelength of the sinusoidal factor 'lambda' in relation to each of the rfsizes
div = np.arange(4, 3.195, -0.05)
v1SimpleCells.v1SimpleCells(orientations, rfsizes, div)
#print(v1SimpleCells.filters)
