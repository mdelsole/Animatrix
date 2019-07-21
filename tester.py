import numpy as np

X = np.random.randn(2,4)
print(X)
np.save('patches.npy', X)

