"""
Input:
    -samples: total number of patches to take
    -winsize: patch width in pixels
    -directory: directory in which the images are stored
Output:
    -X: the image patches as column vectors
"""
import os
import numpy as np
import cv2


def sampleImage(samples, winSize, directory):
    # List all image files in the folder
    fileList = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # Number of images
    numImages = len(fileList)
    # How many samples to take based on number of images
    getSample = round(samples / numImages)

    samples = getSample * numImages

    # Intialize the matrix to hold the patches
    X = np.zeros((winSize**2, samples), "float")

    sampleCounter = 0
    for i in range(numImages):
        # Load the image, convert to grayscale
        img = cv2.imread(os.path.join(directory, fileList[i]), cv2.IMREAD_GRAYSCALE)
        # Transform to double
        img.astype(float)

        # Sample patches in random locations
        width = np.size(img, 1)
        height = np.size(img, 0)
        posx = np.floor(np.random.rand(1, getSample) * (width - winSize))
        posy = np.floor(np.random.rand(1, getSample) * (height - winSize))
        for j in range(getSample):
            X[:, sampleCounter] = np.reshape(img[int(posy[0,j]):(int(posy[0,j]) + winSize),
                                             int(posx[0,j]):(int(posx[0,j]) + winSize)], (winSize**2))
            sampleCounter += 1

    return X