import numpy as np
x = np.random.randint(1,8, size=(8, 4))
print(x)
print(x[0:5,0])
y = np.random.randint(1,8, size=(5,1))
print("Y: ", y)
print(np.reshape(y,(5)))
x[0:5,0] = np.reshape(y,(5))
print(x)

