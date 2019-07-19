import numpy as np


# Utility method for creating Gabor filters

def gaborFilter(size, wavelength, orientation):

    """
    Create a single gabor filter
    Inputs:
        -Size: The size of the filter. Filters are square, so filter will be [size x size]
        -Wavelength: The wavelength of the grating of the filter. Determines tightness of the filter
        -Orientation: The orientation angle (0, 45, 90, -45) of the filter

    Returns:
        -Filt: numpy array of the filter weights, shape = (size, size)

    Parameters:
        -Lambda: Width of the stripes. Increase to produce thicker stripes.
        -Sigma: Bandwidth. Increase to fit to filter size. Decrease to allow more stripes, as each one becomes thinner
        -Gamma: Height. We won't need to change
        -Theta: Orientation (0, 45, 90)

    All parameters are set based of the filter size, as they intent is to fit them correctly to each size
    """

    # Parameters
    lmbda = size * 2. / wavelength
    sigma = lmbda * 0.8
    gamma = 0.3  # spatial aspect ratio: 0.23 < gamma < 0.92
    theta = np.deg2rad(orientation + 90)

    # Generate Gabor filter

    # Create the grid that we will plot the gabor filter on. X, Y become our grid points
    x, y = np.mgrid[:size, :size] - (size // 2)

    # X0 and Y0 of the Gabor Function
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    # The Gabor Function
    filt = np.exp(-(rotx ** 2 + gamma ** 2 * roty ** 2) / (2 * sigma ** 2))
    filt *= np.cos(2 * np.pi * rotx / lmbda)
    filt[np.sqrt(x ** 2 + y ** 2) > (size / 2)] = 0

    # Normalize the filter
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))

    return filt
