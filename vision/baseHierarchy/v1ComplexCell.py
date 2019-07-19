import numpy as np
from torch import nn
import torch

"""
V1's complex cells. These cells pool over inflowing simple cells from the same orientation selectivity, but at
slightly different positions and scales to increase the cells' tolerance.

Complex cells perform a max operation, i.e. for each complex cell the cell in its inflow that fires most is chosen.
In other words, the response y of a complex unit corresponds to the response of the strongest of its afferents.
"""
