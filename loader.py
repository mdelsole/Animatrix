import pickle
import numpy as np

np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

# View whole array, for debugging purposes
from matplotlib import gridspec

# Visualize each filter
spec = gridspec.GridSpec(nrows=1, ncols=4)
fig = plt.figure()

pickle_in = open("output.pkl", "rb")
example_dict = pickle.load(pickle_in)
# Visualize it
for i in range(4):
    ax = fig.add_subplot(spec[0,i])
    ax.imshow(np.reshape(example_dict['s1'][15][7][i], (250, 250)), cmap='Greys_r')
    ax.set_axis_off()

plt.draw()
plt.savefig('RFs.png', bbox_inches='tight')


spec = gridspec.GridSpec(nrows=1, ncols=1)
fig = plt.figure()

reconstructedImage = np.zeros((250,250))

for i in range(4):
    reconstructedImage[np.where(example_dict['s1'][15][3][i]>0.01)] = example_dict['s1'][15][3][i][np.where(example_dict['s1'][15][3][i]>0.01)]

# Visualize it
ax = fig.add_subplot(spec[0,0])
ax.imshow(np.reshape(reconstructedImage, (250,250)), cmap='Greys_r')
ax.set_axis_off()

plt.draw()
plt.savefig('reconstruction.png', bbox_inches='tight')