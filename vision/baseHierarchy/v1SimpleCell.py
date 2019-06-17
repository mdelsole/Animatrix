import numpy as np
from torch import nn
import torch


class v1SimpleCell(nn.Module):

    """
    V1 simple cells are the bottom of the hierarchy. They are exposed to the raw pixel data of the image. Each v1 simple
    cell is a Gabor filter, which detects edges in a certain orientation

    Implemented using PyTorch Conv2d; each channel is loaded with a Gabor filter at a specific orientation

    Inputs:
        -Size: The size of the filter. Filters are square, so filter will be [size x size]
        -Wavelength: The wavelength of the grating of the filter. Determines tightness of the filter
        -Orientation: The orientation angle (0, 45, 90, -45) of the filter
    """

    def __init__(self, size, wavelength, orientations=[90, -45, 0, 45]):
        super().__init__()
        self.numOrientations = len(orientations)
        self.size = size

        # Use PyTorch's Conv2d as a base object. Each "channel" will be an orientation.
        # nn.Conv2d(in channels, out channels, filter size)
        # In channels = the image, Out channels = each filter orientation (produce a filtered image for each filter)
        self.gabor = nn.Conv2d(1, self.numOrientations, size, padding=size // 2, bias=False)

        # Fill the Conv2d filter weights with Gabor filters, one for each orientation
        # for index, value in array
        for i, orientation in enumerate(orientations):
            self.gabor.weight.data[i, 0] = torch.Tensor(gaborFilter(size, wavelength, orientation))

        # A convolution layer filled with ones. This is used to normalize the result in the forward method.
        self.uniform = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # Since everything is pre-computed, no gradient is required
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, img):
        # Apply Gabor filters, take absolute value (for contrast invariance)
        s1_output = torch.abs(self.gabor(img))

        # Normalize
        norm = torch.sqrt(self.uniform(img ** 2))
        norm.data[norm == 0] = 1  # Avoid divide by zero
        s1_output /= norm

        return s1_output


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
