import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core.structure import Structure
from topwave.spinwave import Model, Spec
from topwave.topology import WCC_evolution

#%%
# lattice parameters
lat = [[1.0, 0.0, 0.], [-0.5, np.sqrt(3.0) / 2.0, 0.], [0., 0., 10]]
SG = 191
struc = Structure.from_spacegroup(SG, lat, ['C'], [[1 / 3, 2 / 3, 0.]])

# Construct a Model instance
hc = Model(struc)

# and find all nearest- and next-nearest neighbors
# we enforce P1 (no) symmetry to put DM manually for each bond
hc.generate_couplings(1.1, 1)
hc.show_couplings()

# set the ground state to Ferromagnetic
hc.magnetize([[0, 0, 1], [0, 0, 1]])

# we also put a weak external magnetic field to make the Hamiltonian positive definite
h = 0.01
hc.external_field([0, 0, h])

# assign a isotropic heisenberg exchange to all nearest- and next-nearest neighbors
J = -1
J_NNN = -0#.1
for _ in range(3):
    hc.assign_exchange(J, _)
for _ in range(3, 9):
    hc.assign_exchange(J_NNN, _)

# generate a k-path
ks = np.linspace([0., 0., 0.2], [0.5, 0.5, 0.2], 100)

# solve for the spectrum and plot the 'iconic graphene dispersion relation'
spec_noDM = Spec(hc, ks)

fig, ax = plt.subplots()
for band in spec_noDM.E.T:
    ax.plot(np.linspace(0., 0.5, len(ks)), band, c='blue')
ymin, ymax = ax.get_ylim()
ax.set_ylim([0., ymax])
ax.set_ylabel('Energy [eV]')
ax.set_xlabel(r'$k_x$, $k_y$')

lines = [Line2D([0], [0], color='blue')]
labels = ['w/o DM']
ax.legend(lines, labels)

# put DM interaction on all the next-nearest neighbor bonds to open gap at K
# we have to do it such that the sign of the DM term stays constants around
# one plaquette (check the coupling table!)
D = -0.1
for _ in [3, 4, 5, 7]:
    hc.assign_DM([0, 0, D], _)
for _ in [6, 8]:
    hc.assign_DM([0, 0, -D], _)

hc.show_couplings()

# solve for the spectrum and plot
spec_DM = Spec(hc, ks)
for band in spec_DM.E.T:
    ax.plot(np.linspace(0., 0.5, len(ks)), band, c='red', alpha=0.5)
lines.append(Line2D([0], [0], color='red', alpha=0.5))
labels.append('w/ DM')
ax.legend(lines, labels)
plt.show()

#%%
# get k-points in the whole (2D) BZ
nk = 31
kmin = -0.5
kmax = 0.5
ks = np.linspace(kmin, kmax, nk, endpoint=False)
kx, ky = np.meshgrid(ks, ks, indexing='ij')
loops = np.array([kx.T, ky.T, np.zeros((nk, nk))]).swapaxes(0, 2)

# calculate the Wannier Charge center on each loop for the lower band
occ = [1]
evol = WCC_evolution(hc, loops, occ, test=False)

# plot the evolution of the charge centers
fig, ax = plt.subplots()
for band in evol.WCCs:
    ax.scatter(ks, band)
ax1.set_xlabel(r'$k_x$')
plt.show()


# %%
# get k-points in the whole (2D) BZ
nk = 31
kmin = -0.5
kmax = 0.5
ks = np.linspace(kmin, kmax, nk)
kx, ky = np.meshgrid(ks, ks, indexing='ij')
ks = np.array([kx.flatten(), ky.flatten(), np.zeros(nk ** 2)], dtype=float).T

# compute the spectrum and the Berry Curvature
spec_BC = Spec(hc, ks)
spec_BC.get_berry_curvature()

# plot it
band = 0
fig, ax = plt.subplots()
im = ax.imshow(spec_BC.OMEGA[:, 2, band].reshape((nk, nk)), origin='lower', extent=[kmin, kmax, kmin, kmax])
plt.colorbar(im)
plt.show()

# %%
# parameterize a sphere around the crossing with loops of constant elevation angle
c = [1 / 3, 1 / 3, 0]
r = 0.02
numtheta = 15
numphi = 50
# generating the elevation angles (omitting pole and equator)
thetas = np.linspace(-np.pi, 0, numtheta + 1, endpoint=False)[1:]
# generating azimuthal angles (making sure start and end point are the same)
phis = np.linspace(0, 2 * np.pi, numphi, endpoint=False)
#phis[-1] = 0
loops = []
for theta in thetas:
    x = c[0] + r * np.sin(theta) * np.cos(phis)
    y = c[1] + r * np.sin(theta) * np.sin(phis)
    z = c[2] + r * np.repeat(np.cos(theta), numphi)
    ks = np.array([x, y, z], dtype=float).T
    loops.append(ks)

# calculate the Wannier Charge center on each loop for the lower band
occ = [0]
evol1 = WCC_evolution(hc, loops, occ, test=False)
evol2 = WCC_evolution(hc, loops, occ, test=True)

# plot the evolution of the charge centers
fig, (ax1, ax2) = plt.subplots(1, 2)
for band1, band2 in zip(evol1.WCCs, evol2.WCCs):
    ax1.scatter(thetas, band1)
    ax2.scatter(thetas, band2)
ax1.set_xlabel(r'$\theta$')
ax2.set_xlabel(r'$\theta$')
plt.show()

# %%

# calculate the Berry Curvature on a sphere around K
ks = np.array(loops).reshape((-1, 3))
spec_BC = Spec(hc, ks)
spec_BC.get_berry_curvature_test()

# calculate the Berry Flux (and reshape it again into slices of constant elevation angle)
dS = (ks - c)/r
F = np.einsum('kin, ki ->kn', spec_BC.OMEGA, dS).reshape((numtheta, numphi, -1))

# integrate it over the sphere
C = np.trapz(np.trapz(F, phis, axis=1).T * np.sin(thetas) * np.square(r), thetas, axis=1)

band = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.quiver(*ks.T, *np.ones(ks.T.shape), normalize=True)
ax.scatter(*ks.T)
plt.show()
