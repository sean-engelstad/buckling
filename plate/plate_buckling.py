import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import eigh
import os
from mpl_toolkits.mplot3d import Axes3D

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
nx = 10
ny = 10

# bcs - simply supported on all boundaries
bcs = []
# loop over nodes
I = 0
for iy in range(ny+1):
    for ix in range(nx+1):
        if ix == 0 or ix == nx or iy == 0 or iy == ny:
            # recall dof in [w,dw/dx, dw/dy] dof at each node
            # simply supported constraint w=0 
            bcs += [3*I]
        # dw/dy = 0 on all nodes
        # bcs += [3*I+2]
        if ix == 0 or ix == nx:
            # dw/dy = 0 on x bndry
            bcs += [3*I+2]
        if iy == 0 or iy == ny:
            # dw/dx = 0 on y bndry
            bcs += [3*I+1]
        I += 1

# compute the case names
load_list = np.array([Nx, Nxy, Ny])
nonzero_mask = load_list != 0
if np.sum(nonzero_mask) == 1:
    if Nx != 0:
        case = "Nx"
    elif Nxy != 0:
        case = "Nxy"
    else:
        case = "Ny"
else:
    case = "mixed"

print(f"bcs = {bcs}")
# exit()

# 3 per node (u,ux,uy)
Ndof = 3*(nx+1)*(ny+1)

K = np.zeros((Ndof, Ndof))
G = np.zeros((Ndof,Ndof))

# use hermite cubic interpolation functions
# and complex-step to find the derivatives
def hermite_cubic_functions(imod,a,b,xi_i,eta_i):
    if imod == 0:
        phii = lambda xi,eta : 0.125 * (1 + xi*xi_i)*(1 + eta*eta_i)*(2 + xi*xi_i + eta*eta_i - xi**2 - eta**2)
        dphi_dx = lambda xi,eta : 1.0/a * phii(xi,eta) * ( xii /(1+xi*xi_i) + (xi_i - 2 * xi) / (2 + xi*xi_i + eta*eta_i - xi**2 - eta**2))
        dphi_dy = lambda xi,eta : 1.0/b * phii(xi,eta) * (etai / (1 + eta*eta_i) + (eta_i - 2*eta) / (2 + xi*xi_i + eta*eta_i - xi**2 - eta**2))
    elif imod == 1:
        phii = lambda xi,eta : a/8.0 * xii * (xi * xii + 1)**2 * (xi * xi_i - 1) * (eta * eta_i + 1)
        dphi_dx = lambda xi,eta : 1.0/a * phii(xi,eta) * (2 * xi_i / (xi * xi_i + 1) + xi_i / (xi * xi_i - 1))
        dphi_dy = lambda xi,eta : 1.0/b * phii(xi,eta) * etai / (eta * eta_i + 1)
    elif imod == 2:
        phii = lambda xi,eta : b/8.0 * etai * (xi * xii + 1) * (eta * eta_i + 1)**2 * (eta * eta_i - 1)
        dphi_dx = lambda xi,eta : 1.0/a * phii(xi,eta) * xi_i / (xi * xi_i + 1)
        dphi_dy = lambda xi,eta : 1.0/b * phii(xi,eta) * ( 2 * etai / (eta * eta_i + 1) + etai / (eta * eta_i - 1))
    dphi_dx2 = lambda xi,eta : 1.0/a * np.imag(dphi_dx(xi+1e-30*1j,eta))/1e-30
    dphi_dy2 = lambda xi,eta : 1.0/b * np.imag(dphi_dy(xi,eta+1e-30*1j))/1e-30
    lapl_phi = lambda xi,eta : dphi_dx2(xi,eta) + dphi_dy2(xi,eta)
    return phii, dphi_dx, dphi_dy, dphi_dx2, dphi_dy2,lapl_phi

# plot all 12 hermite cubic interpolation functions using a contour plot
xi = np.linspace(-1,1,50)
eta = np.linspace(-1,1,50)
XI,ETA = np.meshgrid(xi,eta)

if not os.path.exists('hermite-cubic'):
    os.mkdir('hermite-cubic')

ct = 0
xii_list = []
etai_list = []
for inode in range(4):
    if inode in [0,3]:
        xii = -1
    else:
        xii = 1
    if inode in [0,1]:
        etai = -1
    else:
        etai = 1
    for i in range(3):
        xii_list += [xii]
        etai_list += [etai]
        phi,_,_,_,_,_ = hermite_cubic_functions(i,1,1,xii,etai)
        PHI = phi(XI,ETA)
        fig, ax = plt.subplots()
        surf = ax.contourf(XI, ETA, PHI, cmap=cm.coolwarm, antialiased=False)
        ct += 1
        plt.title(f"phi{ct}(xi,eta)")
        plt.savefig(f"hermite-cubic/phi{ct}.png",dpi=400)
        plt.close('all')

def element_matrices(a,b):
    Kelem = np.zeros((12,12))
    Gelem = np.zeros((12,12))
    for i in range(12):
        _, dphii_dx, dphii_dy, _, _, lapl_phii = \
            hermite_cubic_functions(i % 3,a,b,xii_list[i],etai_list[i])
        for j in range(12):
            _,dphij_dx, dphij_dy, _, _, lapl_phij = \
                hermite_cubic_functions(j % 3,a,b,xii_list[j],etai_list[j])

            for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                for eta in [-1/np.sqrt(3), 1/np.sqrt(3)]:
                    Kelem[i,j] += D * lapl_phii(xi,eta) * lapl_phij(xi,eta)
                    Gelem[i,j] += Nx * dphii_dx(xi,eta) * dphij_dx(xi,eta) + \
                        Nxy * ( dphii_dx(xi,eta) * dphij_dy(xi,eta) + dphii_dy(xi,eta) * dphij_dx(xi,eta)) + \
                        Ny * dphii_dy(xi,eta) * dphij_dy(xi,eta)
                    
                    # Nxy * ( dphii_dx(xi,eta) * dphij_dy(xi,eta) + dphii_dy(xi,eta) * dphij_dx(xi,eta))
                    
    #print(f"Kelem = {Kelem}")
    #print(f"Gelem = {Gelem}")
    diag_K = np.diag(Kelem)
    diag_G = np.diag(Gelem)
    print(f"diag K = {diag_K}")
    print(f"diag G = {diag_G}")
    return Kelem,Gelem

for ix in range(nx):
    a = Lx / nx
    for iy in range(ny):
        b = Ly / ny
        # compute element stiffness matrices for each case next
        Kelem,Gelem = element_matrices(a,b)

        for i in range(12):
            inode = np.floor(i/4)
            # compute the global node indices
            IX = ix
            IY = iy
            if not(inode in [0,3]):
                IX += 1
            if not(inode in [0,1]):
                IY += 1
            
            # global node index
            Inode = IX + (ny+1)*IY
            I = 3 * Inode + i % 3

            for j in range(12):
                jnode = np.floor(j/4)
                # compute the global node indices
                JX = ix
                JY = iy
                if not(jnode in [0,3]):
                    JX += 1
                if not(jnode in [0,1]):
                    JY += 1

                # global node index
                Jnode = JX + (ny+1)*JY
                J = 3 * Jnode + j % 3

                K[I,J] += Kelem[i,j]
                G[I,J] += Gelem[i,j]
                        

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
#exit()

#print(f"Kr = {Kr}")
#print(f"Gr = {Gr}")

diag_Kr = np.diag(Kr)
diag_Gr = np.diag(Gr)
print(f"diag Kr = {diag_Kr}")
print(f"diag Gr = {diag_Gr}")
P, U = eigh(Kr, Gr)

if not os.path.exists(case):
    os.mkdir(case)

# plot only the lowest mode shape first
for i in range(3):
    Pi = P[i]
    Ui = U[:,i]

    u_full = np.zeros((Ndof,))
    u_full[rem_dof] = Ui

    # make a surface plot of the displacements
    x = [Lx*_/nx for _ in range(nx+1)]
    y = [Ly*_/ny for _ in range(ny+1)]
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
    fig, ax = plt.subplots()
    surf = ax.contourf(X, Y, W, cmap=cm.coolwarm,
                        antialiased=True)
    ax.set_aspect('equal')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(case + f"/buckling-mode{i+1}.png",dpi=400)
    plt.close('all')