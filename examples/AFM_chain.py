import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymatgen.core.structure import Structure
from topwave.spinwave import Model, Spec
from topwave.topology import WCC_evolution

#%%
# create a chain with a two atom unit cell
a = 1. # lattice constant
SG = 1 # P1 symmetry (no symmetries)
lat = [[a, 0, 0], [0, 10, 0], [0, 0, 10]] # 1D lattice (put vacuum in y- and z-direction)
struc = Structure.from_spacegroup(SG, lat, ['Co', 'Co'], [[0., 0., 0.], [0.5, 0, 0]])

# create a topwave Model and generate the couplings
model = Model(struc)
model.generate_couplings(a/2, SG)
model.show_couplings()

# now we assign AFM coupling to the two next-nearest neighbor hoppings
J = -1 # strength of the Heisenberg Exchange
model.assign_exchange(J, 0)
model.assign_exchange(J, 1)

# put the model in an AFM ground state
ground_state = [[0, 0, 1], [0, 0, 1]]
model.magnetize(ground_state)

# calculate its dispersion relation
ks = np.linspace([0, 0, 0], [1., 0, 0], 101)
spec = Spec(model, ks)

# plot the spectrum
fig, ax = plt.subplots()
for band in spec.E.T:
    ax.plot(ks[:, 0], band, c='k')
ymin, ymax = ax.get_ylim()
ax.set_ylim([0., ymax])
ax.set_xlabel('kx')
ax.set_ylabel('Energy')
plt.show()
