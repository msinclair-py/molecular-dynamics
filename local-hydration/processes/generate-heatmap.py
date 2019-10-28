#!/USR/ENV/BIN PYTHON
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

dat = np.genfromtxt('../Data/periplasm_density.txt', delimiter=' ')
X_dat = dat[:,0]
Y_dat = dat[:,1]
Z_dat = dat[:,2]

X, Y, Z = np.array([]), np.array([]), np.array([])
for i in range(len(X_dat)):
        X = np.append(X, X_dat[i])
        Y = np.append(Y, Y_dat[i])
        Z = np.append(Z, Z_dat[i])

xi = np.linspace(X.min(), X.max(), 1000)
yi = np.linspace(Y.min(), Y.max(), 1000)

zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear')

zmin = 20
zmax = 80
zi[(zi<zmin) | (zi>zmax)] = None

CS = plt.contourf(xi, yi, zi, 10, cmap=plt.cm.cool, vmax=zmax, vmin=zmin)

plt.colorbar()
plt.show()
