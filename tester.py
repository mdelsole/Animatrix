import numpy as np

X = np.random.randint(1,4,(3,3,3))
print(X)
X = X[0:2, :, :]
print(X)

"""
x = [1 2 3 7; 3 4 8 12; 8 9 10 5]
y = [3 4 5 6; 3 9 10 11; 8 9 4 5]
z = cat(3, x, y)


size(x)
%(row num, apply to each col, apply to each matrix(3d))
z(3,:,:) = []

size(x)
"""

