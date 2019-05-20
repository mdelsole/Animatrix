import numpy as np
import scipy.io as sio
np.set_printoptions(threshold=np.nan)

# Defining 8 scale bands
mat_contents = sio.loadmat('/Users/Michael/Documents/Animatrix/universal_patch_set.mat')
print(mat_contents)