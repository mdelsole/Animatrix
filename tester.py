import numpy as np

# Defining 8 scale bands
c1Scale = np.array([2, 2, 3, 0, 1, 2, 3, 4, 5])
print(c1Scale)
c1Scale[np.where(c1Scale == 0)] = 1
print(c1Scale)
