# Preprocess the inputted image

import cv2
from scipy.io import loadmat
from torch import nn
import numpy as np

# Import other classes
from util import normalize, sampleImage, pca
from vision.algorithms import ica
from vision import visualizer
from vision.baseHierarchy import v1ComplexCell

np.set_printoptions(threshold=np.nan)

path = "/Users/Michael/Documents/Animatrix/Opencountry/"
pathFilters = "/Users/Michael/Documents/Animatrix/vision/filters/"

######## V1 Simple Cells ########

print("Learning V1 Simple Cells")

# Parameters
patchsz1= 10
nbases1 = 36
samplesize1 = 50000

print("Sampling Data...")
# Fill X with 50000ish patches
X = sampleImage.sampleImage(samplesize1, patchsz1, path)

print("Normalizing Data...")
X = normalize.normalize(X)


print("Doing PCA Dimension Reduction and Whitening Data...")
# Apply PCA
V1 = pca.pca(X, nbases1)
# Whiten Data
Z = np.matmul(V1, X)

print("Starting ICA...")
w1pca = ica.ica(Z, nbases1)
# Transform back to original space from whitened space
W1 = np.matmul(w1pca, V1)
print(np.shape(V1))
print(np.shape(w1pca))
# Compute A using pseudoinverse (inverting canonical preprocessing is tricky)
A1 = np.linalg.pinv(W1)
np.save('S1.npy', A1)

# visualizer.visualize(A1, 6, 6)

######## V1 Complex Cells ########

# Parameters
ratio1 = 3
minImgSize = 120

print("Learning V1 Complex Cells")
C1Inputs = np.load('S1.npy')
v1ComplexCell.v1ComplexCell(path, pathFilters, np.transpose(w1pca), V1, ratio1, minImgSize)

######## V2 Simple Cells ########






