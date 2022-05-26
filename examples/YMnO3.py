import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.cif import CifWriter
from topwave.model import Model, Spec
from topwave.topology import WCC_evolution

#%%
# build the pymatgen structure (we choose the conventional unit cell)
a = 6.2
b = a
c = 11.4
lat = Lattice.from_parameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=120)
SG = 185
struc = Structure.from_spacegroup(SG, lat, ['Mn'],
                                  [[1/3., 0, 0]])

# Construct a Model instance
model = Model(struc)

model.generate_couplings(6.1, SG)
model.show_couplings()

# assign exchange couplings
J1 = 1
J2 = 1
J3 = -0. # FM interplane coupling
model.set_coupling(J1, 0)
model.set_coupling(J2, 1)
model.set_coupling(J3, 2)
model.set_coupling(J3, 3)

# set ground state
mu = 3.143

spin5 = spin4 = [1, 0, 0]
spin3 = spin2 = [0, 1, 0]
spin1 = spin0 = [-1, -1, 0]
dirs = np.vstack((spin0, spin1, spin2, spin3, spin4, spin5))

model.set_moments(dirs, [mu] * 6)

#CifWriter(struc, write_magmoms=True).write_file('test.mcif')


# %%

# calculate magnon spectrum
nk = 100
delta = 0.0001
ks = np.linspace([0.5, 0.5, 0.5], [0.5, 0.2, 0.5], nk)
spec = Spec(model, ks)
#%%
# plot it
fig, ax = plt.subplots()
for band in spec.E.T:
    ax.plot(band, c='k')
ymin, ymax = ax.get_ylim()
ax.set_ylim([0., ymax])
plt.show()
