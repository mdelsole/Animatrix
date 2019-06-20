import numpy as np
from torch import nn
import torch

"""
V1's complex cells. These cells pool over inflowing simple cells from the same orientation selectivity, but at
slightly different positions and scales to increase the cells' tolerance.

Complex cells perform a max operation, i.e. for each complex cell the cell in its inflow that fires most is chosen.
In other words, the response y of a complex unit corresponds to the response of the strongest of its afferents.
"""


class v1ComplexCell(nn.Module):
    """
    Each complex cell pools over the simple cells that it is assigned to
    Parameters:
        -Size: The size of the MaxPool2d operation being performed by this C1 layer.
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.local_pool = nn.MaxPool2d(size, stride=size // 2, padding=size // 2)

    def forward(self, s1_outputs):
        # Put all outputs into one array
        s1_outputs = torch.cat([out.unsqueeze(0) for out in s1_outputs], 0)

        # Max pool over all scales
        s1_output, _ = torch.max(s1_outputs, dim=0)

        # Pool over local (c1_space x c1_space) neighbourhood
        return self.local_pool(s1_output)

