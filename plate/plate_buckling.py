import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import eigh

Lx = 1.0
Ly = 1.0
E = 2.0e8
h = 1e-3 # 1 mm thick
nu = 0.3
D = E * h**3 / 12.0 / (1 - nu**2)

# loading type
Nx = 1.0
Nxy = 0.0
Ny = 0.0

# n rect elements in each direction (linear pattern)
nx = 2
ny = 2

# bcs - simply supported on all boundaries
bcs = []
# loop over nodes
I = 0
for iy in range(ny+1):
    for ix in range(nx+1):
        if ix == 0 or ix == nx or iy == 0 or iy == ny:
        # if ix == 0 or ix == nx:
            # only constraint w in [w,dw/dx, dw/dy] dof at each node
            bcs += [3*I+_ for _ in range(1)]
        I += 1

print(f"bcs = {bcs}")
# exit()

# 3 per node (u,ux,uy)
Ndof = 3*(nx+1)*(ny+1)

K = np.zeros((Ndof, Ndof))
G = np.zeros((Ndof,Ndof))

# use hermite cubic interpolation functions
# and complex-step to find the derivatives
def hermite_cubic_functions(imod,a,b,xii,etai):
    if imod == 0:
        phii = lambda xi,eta : 0.125 * (1 + xi*xii)*(1 + eta*etai)*(2 + xi*xii + eta*etai - xi**2 - eta**2)
        dphii_dx = lambda xi,eta : 1.0/a * ( xii * phii(xi,eta)/(1+xi*xii) + (xii - 2 * xi) * phii(xi,eta) / (2 + xi*xii + eta*etai - xi**2 - eta**2))
        dphii_dy = lambda xi,eta : 1.0/b * phii(xi,eta) * (etai / (1 + eta*etai) + (etai - 2*eta) / (2 + xi*xii + eta*etai - xi**2 - eta**2))
    elif imod == 1:
        phii = lambda xi,eta : a/8.0 * xii * (xi * xii + 1)**2 * (xi * xii - 1) * (eta * etai + 1)
        dphii_dx = lambda xi,eta : 1.0/a * phii(xi,eta) * (2 * xii / (xi * xii + 1) + xii / (xi * xii - 1))
        dphii_dy = lambda xi,eta : 1.0/b * phii(xi,eta) * etai / (eta * etai + 1)
    elif imod == 2:
        phii = lambda xi,eta : b/8.0 * etai * (xi * xii + 1) * (eta * etai + 1)**2 * (eta * etai - 1)
        dphii_dx = lambda xi,eta : 1.0/a * phii(xi,eta) * xii / (xi * xii + 1)
        dphii_dy = lambda xi,eta : 1.0/b * phii(xi,eta) * ( 2 * etai / (eta * etai + 1) + etai / (eta * etai - 1))
    dphii_dx2 = lambda xi,eta : 1.0/a * np.imag(dphii_dx(xi+1e-30*1j,eta))/1e-30
    dphii_dy2 = lambda xi,eta : 1.0/b * np.imag(dphii_dy(xi,eta+1e-30*1j))/1e-30
    lapl_phii = lambda xi,eta : dphii_dx2(xi,eta) + dphii_dy2(xi,eta)
    return phii, dphii_dx, dphii_dy, dphii_dx2, dphii_dy2,lapl_phii

for ix in range(nx):
    a = Lx / nx
    for iy in range(ny):
        b = Ly / ny
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
            Inode = IX + (ny+1)*IY
            I = 3 * Inode + i % 3

            # get the hermite cubic lambda functions
            _, dphii_dx, dphii_dy, _, _, lapl_phii = hermite_cubic_functions(i % 3,a,b,xii,etai)

            # perform the gauss quadrature and find the derivatives as well

            for j in range(12):
                jnode = np.floor(j/4)
                # compute the global node indices
                if jnode in [0,3]:
                    JX = ix
                    xij = -1
                else:
                    JX = ix + 1
                    xij = 1
                if jnode in [0,1]:
                    JY = iy
                    etaj = -1
                else:
                    JY = iy+1
                    etaj = 1

                # global node index
                Jnode = JX + (ny+1)*JY
                J = 3 * Jnode + j % 3

                _,dphij_dx, dphij_dy, _, _, lapl_phij = hermite_cubic_functions(j % 3,a,b,xij,etaj)

                # compute the stiffness and stability matrix coefficients using Gaussian quadrature
                for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                    for eta in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                        K[I,J] += D * lapl_phii(xi,eta) * lapl_phij(xi,eta)
                        G[I,J] += Nx * dphii_dx(xi,eta) * dphij_dx(xi,eta) + \
                            Nxy *( dphii_dx(xi,eta) * dphij_dy(xi,eta) + dphii_dy(xi,eta) * dphij_dx(xi,eta)) + \
                            Ny * dphii_dy(xi,eta) * dphij_dy(xi,eta)
                        

detK = np.linalg.det(K)
detG = np.linalg.det(G)
print(f"detK = {detK}")
print(f"detG = {detG}")

# print(f"K = {K}")
# print(f"G = {G}")

# apply BCs to the problem
full_dof = [_ for _ in range(Ndof)]
rem_dof = [_ for _ in full_dof if not(_ in bcs)]
print(f"rem dof = {rem_dof}")

# solve the reduced / constrained form
# of the eigenvalue problem
Kr = K[rem_dof,:][:,rem_dof]
Gr = G[rem_dof,:][:,rem_dof]

detKr = np.linalg.det(Kr)
detGr = np.linalg.det(Gr)
print(f"detKr = {detKr}")
print(f"detGr = {detGr}")
exit()

#print(f"Kr = {Kr}")
#print(f"Gr = {Gr}")

P, U = eigh(Kr, Gr)

# plot only the lowest mode shape first
for i in range(1):
    Pi = P[i]
    Ui = U[:,i]

    u_full = np.zeros((Ndof,))
    u_full[rem_dof] = Ui

    # make a surface plot of the displacements
    x = [Lx*_/Nx for _ in range(nx+1)]
    y = [Ly*_/Ny for _ in range(ny+1)]
    X,Y = np.meshgrid(x,y)
    w = u_full[::3] # [goes out in x direction first]
    W = np.zeros((nx+1,ny+1))
    ct = 0
    for iy in range(ny+1):
        for ix in range(nx+1):
            W[ix,iy] = w[ct]
            ct += 1

    print(f"P[{i}] = {Pi}")
    print(f"U[{i}] = {w}")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, W, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

plt.savefig("plate-buckling-modes.png", dpi=400)