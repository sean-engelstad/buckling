import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

Lx = 1.0
Ly = 1.0
E = 2.0e8
h = 1e-3 # 1 mm thick
nu = 0.3
D = E * h**3 / 12.0 / (1 - nu**2)


Nx = 10
Ny = 10
# 3 per node (u,ux,uy)
Ndof = 3*(Nx+1)*(Ny+1)

K = np.zeros((Ndof, Ndof))
G = np.zeros((Ndof,Ndof))

# use hermite cubic interpolation functions
# and complex-step to find the derivatives

for ix in range(Nx):
    a = Lx / Nx
    for iy in range(Ny):
        b = Ly / Ny
        # compute element stiffness matrices for each case next
        for i in range(12):
            inode = np.floor(i/4)
            # compute the global node indices
            if inode in [0,3]:
                IX = ix
                xii = -1
            else:
                IX = ix + 1
                xii = 1
            if inode in [0,1]:
                IY = iy
                etai = -1
            else:
                IY = iy+1
                etai = 1
            
            # global node index
            Inode = IX + (Ny+1)*IY
            I = 3 * Inode + i % 3

            # write the hermite cubic interpolation function phii and it's derivatives
            imod = i % 3
            if imod == 0:
                phii = lambda xi,eta : 0.125 * (1 + xi*xii)*(1 + eta*etai)*(2 + xi*xii + eta*etai - xi**2 - eta**2)
            elif imod == 1:
                phii = lambda xi,eta : a/8.0 * xii * (xi * xii + 1)**2 * (xi * xii - 1) * (eta * etai + 1)
            elif imod == 2:
                phii = lambda xi,eta : b/8.0 * etai * (xi * xii + 1) * (eta * etai + 1)**2 * (eta * etai - 1)

            # write lambda expressions for the derivatives
            

            # perform the gauss quadrature and find the derivatives as well

            for j in range(12):
                jnode = np.floor(j/4)
                # compute the global node indices
                if inode in [0,3]:
                    JX = ix
                else:
                    JX = ix + 1
                if inode in [0,1]:
                    JY = iy
                else:
                    JY = iy+1

                # global node index
                Jnode = JX + (Ny+1)*JY
                J = 3 * Jnode + j % 3

                # compute the hermite cubic function 

                # compute the stiffness and stability matrix coefficients using Gaussian quadrature
                K[I,J]