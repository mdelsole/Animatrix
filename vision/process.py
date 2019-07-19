# Preprocess the inputted image

import cv2
from scipy.io import loadmat
from torch import nn
import torch
from util import normalize, sampleImage, pca
from vision.baseHierarchy import v1ComplexCell, v2Cell, v4Cell, v1SimpleCell

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
V1 = pca.pca(X,nbases1)
Z = V1 * X



