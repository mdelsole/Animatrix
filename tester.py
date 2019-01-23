import numpy as np

c1Scale = np.arange(1,18,2)
print(c1Scale)
nBands= np.size(c1Scale)-1
nScales=c1Scale[-1]-1
print(nBands)
print(nScales)