import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
#mpl.use('WebAgg')
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathBase
from pymatgen.symmetry.bandstructure import HighSymmKpath
from topwave.spinwave import Model, Spec
from topwave.topology import WCC_evolution

#%%
# build the pymatgen structure (we choose the conventional unit cell)
a = 5.74900
b = 9.95760
c = 11.88600

lat = [[a, 0, 0], [0, b, 0], [0, 0, c]]
SG = 18
struc = Structure.from_spacegroup(SG, lat, ['Co'], [[0, 1/3, 1/4]])

# Construct a Model instance
model = Model(struc)

# generate the couplings up to next-next-nearest neighbors
model.generate_couplings(9, SG)
model.show_couplings()

# assign couplings that result in the stripy ground state
# phase diagram in: PHYSICAL REVIEW B 91, 180407(R) (2015)
J1 = 1
J2 = 0.4
J3 = -0.4
model.assign_exchange(J2, 0)
model.assign_exchange(J2, 1)
model.assign_exchange(J1, 2)
model.assign_exchange(J1, 3)
model.assign_exchange(J3, 4)
model.assign_exchange(J3, 5)

model.external_field([1, 2, 1])

# put the stripy ground state
gs = [[0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0]]
model.magnetize(gs)


# %%
# calculate the spectrum along some k-path
nk = 100
delta = 0.1
'''
ks = np.concatenate((np.linspace([delta, 0, 0], [0.5, 0, 0], nk, endpoint=False),
                     np.linspace([0.5, 0, 0], [0.5, 0.5-delta, 0], nk, endpoint=False),
                     np.linspace([0.5, 0.5-delta, 0], [delta, 0, 0], nk, endpoint=False),
                     np.linspace([delta, 0, 0], [0, 0, 0.5], nk, endpoint=False),
                     np.linspace([0, 0, 0.5], [0.5, 0, 0.5], nk, endpoint=False),
                     np.linspace([0.5, 0, 0.5], [0.5, 0.5, 0.5], nk, endpoint=False),
                     np.linspace([0.5, 0.5, 0.5], [delta, delta, delta], nk, endpoint=True)))
'''
ks = np.linspace([0, 0, 0.5], [0.5, 0, 0.5], nk)
spec = Spec(model, ks)

# plot it
fig, ax = plt.subplots()
for band in spec.E.T:
    ax.plot(band, c='k')
ymin, ymax = ax.get_ylim()
ax.set_ylim([0, ymax])
plt.show()

# %%
# 3D band structure
nk = 21
kz = 0.0
ks = np.linspace(-0.5, 0.5, nk)
kx, ky = np.meshgrid(ks, ks, indexing='ij')
ks = np.array([kx.flatten(), ky.flatten(), np.zeros(nk ** 2) + kz], dtype=float).T

spec = Spec(model, ks)

# plot it
band = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(ks[:, 0].reshape((nk, nk)), ks[:, 1].reshape((nk, nk)), spec.E[:, band].reshape((nk, nk)),
                cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_surface(ks[:, 0].reshape((nk, nk)), ks[:, 1].reshape((nk, nk)), spec.E[:, band + 2].reshape((nk, nk)),
                cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(elev=10)
plt.show()



