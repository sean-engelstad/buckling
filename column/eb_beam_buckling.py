import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# problem inputs
L = 1.0
EI = 1.0

n_elem = 20
n_nodes = n_elem + 1
n_dof = 2 * n_elem + 2
he = L / n_elem # element length (like dx)

# stiffness matrix K and stability matrix G
K = np.zeros((n_dof, n_dof))
G = np.zeros((n_dof, n_dof))

# loop over elements to compute full stiffness and stability matrices
for ielem in range(n_elem):
    K[2*ielem:2*ielem+4,2*ielem:2*ielem+4] += 2*EI/he**3 * np.array([
        [6, -3*he, -6, -3*he],
        [-3*he, 2*he**2, 3*he, he**2],
        [-6, 3*he, 6, 3*he],
        [-3*he, he**2, 3*he, 2*he**2]
    ])

    G[2*ielem:2*ielem+4,2*ielem:2*ielem+4] += 1.0/30.0/he * np.array([
        [36, -3*he, -36, -3*he],
        [-3*he, 4*he**2, 3*he, -he**2],
        [-36, 3*he, 36, 3*he],
        [-3*he, -he**2, 3*he, 4*he**2]
    ])

print(f"K = {K}")
print(f"G = {G}")

# apply BCs to the problem
    # pinned-pinned BCs
full_dof = [_ for _ in range(n_dof)]
# remember element dof are [uL, thL, uR, thR]
bcs = [0, n_dof-2]
rem_dof = [_ for _ in full_dof if not(_ in bcs)]
print(f"rem dof = {rem_dof}")

# solve the reduced / constrained form
# of the eigenvalue problem
Kr = K[rem_dof,:][:,rem_dof]
Gr = G[rem_dof,:][:,rem_dof]

#print(f"Kr = {Kr}")
#print(f"Gr = {Gr}")

P, U = eigh(Kr, Gr)

for i in range(3):
    Pi = P[i]
    Ui = U[:,i]

    u_full = np.zeros((n_dof))
    u_full[rem_dof] = Ui

    x = [he*i for i in range(n_nodes)]
    ux = u_full[::2]

    print(f"P[{i}] = {Pi}")
    print(f"U[{i}] = {ux}")
    plt.plot(x,ux, label=f"mode-{i}")

plt.legend()
plt.savefig("eb-buckling-modes.png", dpi=400)

# error in lowest eigenvalue
# exact is Pcr = pi^2 * EI/L^2 for simply supported
# this is the Euler's buckling load for a column
Pcr_euler = np.pi**2 * EI / L**2
err = (P[0] - Pcr_euler) / Pcr_euler
print(f"rel error in Pcr = {err}")