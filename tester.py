import numpy as np
from util import im2col

x = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10], [8, 9, 10, 11]])
print(x)
z =im2col.im2col_distinct(x, [2, 4])

print(z)