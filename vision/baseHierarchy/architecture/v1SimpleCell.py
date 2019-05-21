import numpy as np
from torch import nn
import torch


class v1SimpleCell(nn.Module):

    """A layer of S1 units with different orientations but the same scale.
        The S1 units are at the bottom of the network. They are exposed to the raw
        pixel data of the image. Each S1 unit is a Gabor filter, which detects
        edges in a certain orientation. They are implemented as PyTorch Conv2d
        modules, where each channel is loaded with a Gabor filter in a specific
        orientation.
        Parameters
        ----------
        size : int
            The size of the filters, measured in pixels. The filters are square,
            hence only a single number (either width or height) needs to be
            specified.
        wavelength : float
            The wavelength of the grating in the filter, relative to the half the
            size of the filter. For example, a wavelength of 2 will generate a
            Gabor filter with a grating that contains exactly one wave. This
            determines the "tightness" of the filter.
        orientations : list of float
            The orientations of the Gabor filters, in degrees.
        """

    def __init__(self, size, wavelength, orientations=[90, -45, 0, 45]):
        super().__init__()
        self.num_orientations = len(orientations)
        self.size = size

        # Use PyTorch's Conv2d as a base object. Each "channel" will be an
        # orientation.
        self.gabor = nn.Conv2d(1, self.num_orientations, size,
                               padding=size // 2, bias=False)

        # Fill the Conv2d filter weights with Gabor kernels: one for each
        # orientation
        for channel, orientation in enumerate(orientations):
            self.gabor.weight.data[channel, 0] = torch.Tensor(
                gaborFilter(size, wavelength, orientation))

        # A convolution layer filled with ones. This is used to normalize the
        # result in the forward method.
        self.uniform = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # Since everything is pre-computed, no gradient is required
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, img):
        """Apply Gabor filters, take absolute value, and normalize."""
        s1_output = torch.abs(self.gabor(img))
        norm = torch.sqrt(self.uniform(img ** 2))
        norm.data[norm == 0] = 1  # To avoid divide by zero
        s1_output /= norm
        return s1_output

def gaborFilter(size, wavelength, orientation):
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
