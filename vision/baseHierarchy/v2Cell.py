import numpy as np
from torch import nn
import torch

class v2Cell(nn.Module):
    """
    The activation of v2 cells is computed by taking the distance between
    the output of the v1 complex cells below and a set of predefined patches. This
    distance is computed as:
      d = sqrt( (w - p)^2 )
        = sqrt( w^2 - 2pw + p^2 )
    Inputs:
        -Patches: Numpy array, the precomputed patches to lead into the weights of this layer
            -Shape = (n_patches, n_orientations, size, size)
        -Activation: Which activation function to use for the units
            -'gaussian' or 'euclidean'
        -Sigma: Float, the sharpness of the tuning (sigma in eqn 1 of [1]_). Defaults to 1.
    """
    def __init__(self, patches, activation='gaussian', sigma=1):
        super().__init__()

        # Parameters
        self.activation = activation
        self.sigma = sigma

        # Get parameters from the patches
        num_patches, num_orientations, size, _ = patches.shape

        # Main convolution layer
        # Conv2d(
        self.conv = nn.Conv2d(in_channels=num_orientations, out_channels=num_orientations * num_patches,
                              kernel_size=size, padding=size // 2, groups=num_orientations, bias=False)
        self.conv.weight.data = torch.Tensor(patches.transpose(1, 0, 2, 3).reshape(1600, 1, size, size))

        # A convolution layer filled with ones. This is used for the distance computation
        self.uniform = nn.Conv2d(1, 1, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # This is also used for the distance computation
        self.patches_sum_sq = nn.Parameter(torch.Tensor((patches ** 2).sum(axis=(1, 2, 3))))

        self.num_patches = num_patches
        self.num_orientations = num_orientations
        self.size = size

        # No gradient required for this layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, c1_outputs):
        s2_outputs = []
        for c1_output in c1_outputs:
            conv_output = self.conv(c1_output)

            # Unstack the orientations
            conv_output_size = conv_output.shape[3]
            conv_output = conv_output.view(-1, self.num_orientations, self.num_patches, conv_output_size,
                                           conv_output_size)

            # Pool over orientations
            conv_output = conv_output.sum(dim=1)

            # Compute distance
            c1_sq = self.uniform(torch.sum(c1_output ** 2, dim=1, keepdim=True))
            dist = c1_sq - 2 * conv_output
            dist += self.patches_sum_sq[None, :, None, None]

            # Apply activation function
            if self.activation == 'gaussian':
                dist = torch.exp(- 1 / (2 * self.sigma ** 2) * dist)
            elif self.activation == 'euclidean':
                dist[dist < 0] = 0  # Negative values shouldn't occur
                torch.sqrt_(dist)
                dist = -dist
            else:
                raise ValueError("activation parameter should be either 'gaussian' or 'euclidean'.")

            s2_outputs.append(dist)
        return s2_outputs
