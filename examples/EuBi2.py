import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core.structure import Structure, Lattice
from topwave.model import Model, Spec
from topwave.topology import WCC_evolution

#%%
# build the pymatgen structure (we choose the conventional unit cell)
a = 4.80980000
b = 17.48930400
c = 4.62743000
lat = Lattice.from_parameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)
SG = 63
struc = Structure.from_spacegroup(SG, lat, ['Bi', 'Bi'],
                                  [[0.000000, 0.437495, 0.750000],
                                   [0.500000, 0.247340, 0.750000]])

# Construct a Model instance
model = Model(struc)

# generate all couplings
maxdist = 3.5
model.generate_couplings(maxdist, SG)
model.show_couplings()

J0 = -1
J1 = -0.4

model.set_coupling(J0, 0)
model.set_coupling(J1, 1)

# put small magnetic field
h = 0.001
B = np.array([0, 0, 1], dtype=float)
model.set_field(h * B)

# put a FM ground state
magmom = 1
gs = magmom * np.tile(B, (8,1))
model.set_moments(gs)

#%%
# calculate spin wave spectrum
nk = 41
ks = np.linspace([0, 0, 0], [0.5, 0.5, 0.5], nk, endpoint=True)

spec = Spec(model, ks)

fig, ax = plt.subplots()
for band in spec.E.T:
    ax.plot(band, c='blue')
ymin, ymax = ax.get_ylim()
ax.set_ylim([0., ymax])
ax.set_ylabel('Energy [eV]')
ax.set_xlabel(r'$k$')
plt.show()



