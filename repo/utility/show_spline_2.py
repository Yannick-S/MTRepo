from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)

b_1  = np.zeros((100))
b_1[20:30] = np.arange(0,1,0.1)
b_1[30:40] = 1 - np.arange(0,1,0.1)

b_2  = np.zeros((100))
b_2[20:30] = np.arange(0,1,0.1)
b_2[30:40] = 1 - np.arange(0,1,0.1)

b_1 = np.matmul(np.ones((100,1)), b_1.reshape(1,100))
b_2 = np.matmul(np.ones((100,1)), b_2.reshape(1,100)).T
Z = b_1 + b_2
Z = Z / Z.max()
# Plot the surface..
cmap = cm.get_cmap('viridis')
surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()