# Preprocess the inputted image

import cv2
from scipy.io import loadmat
from torch import nn
import torch

from vision.baseHierarchy import v1ComplexCell, v2Cell, v4Cell, v1SimpleCell

filename = "/Users/Michael/Documents/Animatrix/testImages/lena.jpg"
path = "/Users/Michael/Documents/Animatrix/vision/baseHierarchy/architecture/connectionArrays/"

print("Processing image "+filename)
# Load the image in grayscaled
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

class visualSystem(nn.Module):

    def __init__(self, universal_patch_set, s2_act='gaussian'):
        super().__init__()

        # Create the v1 simple cells
        # Each "cell" is a layer of cells of all the different orientations at the particular size
        # We adjust the wavelength so it fits nicely with the increase in rfsize
        self.v1SimpleCells = [
            v1SimpleCell.v1SimpleCell(size=7, wavelength=4),
            v1SimpleCell.v1SimpleCell(size=9, wavelength=3.95),
            v1SimpleCell.v1SimpleCell(size=11, wavelength=3.9),
            v1SimpleCell.v1SimpleCell(size=13, wavelength=3.85),
            v1SimpleCell.v1SimpleCell(size=15, wavelength=3.8),
            v1SimpleCell.v1SimpleCell(size=17, wavelength=3.75),
            v1SimpleCell.v1SimpleCell(size=19, wavelength=3.7),
            v1SimpleCell.v1SimpleCell(size=21, wavelength=3.65),
            v1SimpleCell.v1SimpleCell(size=23, wavelength=3.6),
            v1SimpleCell.v1SimpleCell(size=25, wavelength=3.55),
            v1SimpleCell.v1SimpleCell(size=27, wavelength=3.5),
            v1SimpleCell.v1SimpleCell(size=29, wavelength=3.45),
            v1SimpleCell.v1SimpleCell(size=31, wavelength=3.4),
            v1SimpleCell.v1SimpleCell(size=33, wavelength=3.35),
            v1SimpleCell.v1SimpleCell(size=35, wavelength=3.3),
            v1SimpleCell.v1SimpleCell(size=37, wavelength=3.25),
            ]

        # Add the created v1 simple cells to the model
        for cell in self.v1SimpleCells:
            self.add_module('s1_%02d' % cell.size, cell)

        # Create the v1 complex cells
        # Each complex cell layer pools across two simple cell layers
        self.v1ComplexCells = [
            v1ComplexCell.v1ComplexCell(size=8),
            v1ComplexCell.v1ComplexCell(size=10),
            v1ComplexCell.v1ComplexCell(size=12),
            v1ComplexCell.v1ComplexCell(size=14),
            v1ComplexCell.v1ComplexCell(size=16),
            v1ComplexCell.v1ComplexCell(size=18),
            v1ComplexCell.v1ComplexCell(size=20),
            v1ComplexCell.v1ComplexCell(size=22),
        ]

        # Add the created v1 complex cells to the model
        for cell in self.v1ComplexCells:
            self.add_module('c1_%02d' % cell.size, cell)

        # Read the universal patch set for the v2 layer
        m = loadmat(universal_patch_set)
        patches = [patch.reshape(shape[[2, 1, 0, 3]]).transpose(3, 0, 2, 1)
                   for patch, shape in zip(m['patches'][0], m['patchSizes'].T)]

        # One v2 layer for each patch scale, operating on all v1 complex cell layers
        self.v2Cells = [v2Cell.v2Cell(patches=scale_patches, activation=s2_act) for scale_patches in patches]

        # Explicitly add the v2 units as submodules of the model
        for i, cell in enumerate(self.v2Cells):
            self.add_module('s2_%d' % i, cell)

        # One C2 layer operating on each scale
        self.v4Cells = [v4Cell.v4Cell() for cell in self.v2Cells]

        # Explicitly add the C2 units as submodules of the model
        for i, cell in enumerate(self.v4Cells):
            self.add_module('c2_%d' % i, cell)

    def run_all_layers(self, img):
        """
        Compute the activation for each layer.
        Parameters:
            -img: A batch of images to run through the model
                -Tensor, shape = (batch_size, 1, height, width)
        Returns:
            -s1_outputs: List of the output of v1 simple cells at each scale
                -List of Tensors, shape = (batch_size, num_orientations, height, width)
            -c1_outputs: List of the output of v1 complex cells at each scale
                -List of Tensors, shape (batch_size, num_orientations, height, width)
            -s2_outputs: The output of v2 cells both for each C1 scale and for each patch scale
                -List of lists of Tensors, shape = (batch_size, num_patches, height, width)
            -c2_outputs: For each patch scale, the output of the layer of C2 units.
                -List of Tensors, shape = (batch_size, num_patches)
        """
        s1_outputs = [s1(img) for s1 in self.v1SimpleCells]

        # Each C1 layer pools across two S1 layers
        c1_outputs = []

        # Pair a c1 cell to 2 v1 simple cell layers

        # For (each c1 cell, i) in (v1ComplexCells, range(numV1SimpleCells/2)):
        #   c1Output = c1(
        for c1, i in zip(self.v1ComplexCells, range(0, len(self.v1SimpleCells), 2)):
            c1_outputs.append(c1(s1_outputs[i:i + 2]))

        s2_outputs = [s2(c1_outputs) for s2 in self.v2Cells]
        c2_outputs = [c2(s2) for c2, s2 in zip(self.v4Cells, s2_outputs)]

        return s1_outputs, c1_outputs, s2_outputs, c2_outputs

    def forward(self, img):
        """Run through everything and concatenate the output of the C2s."""
        c2_outputs = self.run_all_layers(img)[-1]
        c2_outputs = torch.cat([c2_out[:, None, :] for c2_out in c2_outputs], 1)
        return c2_outputs

    def get_all_layers(self, img):
        """Get the activation for all layers as NumPy arrays.
        Parameters
        ----------
        img : Tensor, shape (batch_size, 1, height, width)
            A batch of images to run through the model
        Returns
        -------
        s1_outputs : List of arrays, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of S1 units.
        c1_outputs : List of arrays, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of C1 units.
        s2_outputs : List of lists of arrays, shape (batch_size, num_patches, height, width)
            For each C1 scale and each patch scale, the output of the layer of
            S2 units.
        c2_outputs : List of arrays, shape (batch_size, num_patches)
            For each patch scale, the output of the layer of C2 units.
        """
        s1_out, c1_out, s2_out, c2_out = self.run_all_layers(img)
        return (
            [s1.cpu().detach().numpy() for s1 in s1_out],
            [c1.cpu().detach().numpy() for c1 in c1_out],
            [[s2_.cpu().detach().numpy() for s2_ in s2] for s2 in s2_out],
            [c2.cpu().detach().numpy() for c2 in c2_out],
        )