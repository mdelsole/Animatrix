# Preprocess the inputted image

import cv2
from scipy.io import loadmat
from torch import nn
import numpy as np
from util import normalize, sampleImage, pca
from vision.algorithms import ica

path = "/Users/Michael/Documents/Animatrix/Opencountry/"

print("Learning V1 Simple Cells")

#Paramaters
patchsz1= 10
nbases1 = 36
samplesize1 = 50000

print("Sampling Data...")
X = sampleImage.sampleImage(samplesize1, patchsz1, path)

print("Normalizing Data...")
X = normalize.normalize(X)

print("Doing PCA Dimension Reduction and Whitening Data...")
V1 = pca.pca(X, nbases1)
Z = np.matmul(V1, X)

print("Starting ICA...")
w1pca = ica.ica(Z, nbases1)
# Transform back to original space from whitened space
W1 = w1pca*V1



