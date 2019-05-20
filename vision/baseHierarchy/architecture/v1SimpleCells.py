import math
import numpy as np
import matplotlib.pyplot as plt

# View whole array, for debugging purposes
from matplotlib import gridspec

np.set_printoptions(threshold=np.nan)

"""

Builds the dictionary of V1 simple cells
This method returns an array of simple Gabor filters. Each Gabor filter is a square 2D array. The inner loop runs through 
orientations (4 orientations per scale) while the outer loop runs through scales

"""


def v1SimpleCells(size, wavelength, orientation):
    """Create a single gabor filter.
        Parameters
        ----------
        size : int
            The size of the filter, measured in pixels. The filter is square, hence
            only a single number (either width or height) needs to be specified.
        wavelength : float
            The wavelength of the grating in the filter, relative to the half the
            size of the filter. For example, a wavelength of 2 will generate a
            Gabor filter with a grating that contains exactly one wave. This
            determines the "tightness" of the filter.
        orientation : float
            The orientation of the grating in the filter, in degrees.
        Returns
        -------
        filt : ndarray, shape (size, size)
            The filter weights.
        """
    lambda_ = size * 2. / wavelength
    sigma = lambda_ * 0.8
    gamma = 0.3  # spatial aspect ratio: 0.23 < gamma < 0.92
    theta = np.deg2rad(orientation + 90)

    # Generate Gabor filter
    x, y = np.mgrid[:size, :size] - (size // 2)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    filt = np.exp(-(rotx ** 2 + gamma ** 2 * roty ** 2) / (2 * sigma ** 2))
    filt *= np.cos(2 * np.pi * rotx / lambda_)
    filt[np.sqrt(x ** 2 + y ** 2) > (size / 2)] = 0

    # Normalize the filter
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))

    return filt
