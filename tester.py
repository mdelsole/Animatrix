import numpy as np

X = np.random.randn(2,4)
print(np.mean(X,0))
print("Shape: ", np.shape(np.mean(X, 0, keepdims=True)))