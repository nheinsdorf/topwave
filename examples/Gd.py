import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core.structure import Structure, Lattice
from topwave.spinwave import Model, Spec
from topwave.topology import WCC_evolution

#%%
# build the pymatgen structure (we choose the conventional unit cell)
a = 3.61393372
c = 5.77007400
lat = Lattice.from_parameters(a=a, b=a, c=c, alpha=90, beta=90, gamma=120)
SG = 194
struc = Structure.from_spacegroup(SG, lat, ['Gd'],
                                  [[0.66666667, 0.33333333, 0.75000000]])


# Construct a Model instance
model = Model(struc)

# generate all couplings
maxdist = 5.5
model.generate_couplings(maxdist, SG)
model.show_couplings()

J0 = -1
J1 = -0.4
J2 = -0.3

model.assign_exchange(J0, 0)
model.assign_exchange(J1, 1)
model.assign_exchange(J2, 2)


# put small magnetic field
h = 0.001
B = np.array([0, 0, 1], dtype=float)
model.external_field(h * B)

# put a FM ground state
magmom = 7/2.
gs = magmom * np.tile(B, (2,1))
model.magnetize(gs)

#%%
# calculate spin wave spectrum
nk = 41
ks = np.linspace([0., 0., 0.], [0.5, 0, 0], nk, endpoint=True)

spec = Spec(model, ks)

fig, ax = plt.subplots()
for band in spec.E.T:
    ax.plot(band, c='blue')
ymin, ymax = ax.get_ylim()
ax.set_ylim([0., ymax])
ax.set_ylabel('Energy [eV]')
ax.set_xlabel(r'$k$')
plt.show()


