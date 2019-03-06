from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


while True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    n = 50
    # Make data.
    X = np.arange(0, 1, 1/(n+1))
    Y = np.arange(0, 1, 1/(n+1))
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros((n+1,n+1))
    for i in range(5-1):
        for j in range(5-1):
            b_1  = np.zeros((n))
            b_1[int(n/5*i):int(n/5*(i+1))] = np.arange(0,1,5/n)
            b_1[int(n/5*(i+1)):int(n/5*(i+2))] = 1 - np.arange(0,1,5/n)

            b_2  = np.zeros((n))
            b_2[int(n/5*j):int(n/5*(j+1))] = np.arange(0,1,5/n)
            b_2[int(n/5*(j+1)):int(n/5*(j+2))] = 1 - np.arange(0,1,5/n)

            w =np.random.rand(1) 
            Z[:n,:n] += np.matmul(b_1.reshape(n,1), b_2.reshape(1,n)) * w

    # Plot the surface.

    cmap = cm.get_cmap('viridis')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()