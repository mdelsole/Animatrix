import scipy.signal as scp
import numpy as np

"""
Given an image and pooling range, returns an image where each "pixel" represents the sums of the pixel values within 
the pooling range of the original pixel.
"""

def sumFilter(imgIn, radius):

    # Size of 4 means it was a vector inputted
    if np.size(radius)== 4:
        imgOut = scp.convolve2d(imgIn,(radius(2)+radius(4)+1,radius(1)+radius(3)+1), mode='same')
    if np.size(radius)== 1:
        imgOut = scp.convolve2d(imgIn,np.ones(2*radius+1), 'same')