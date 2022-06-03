import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.structure import Structure
from topwave.model import TightBindingModel, Spec

#%%

# build the pymatgen structure (we choose the conventional unit cell)
a = 4.925866
b = a
c = a
sg = 198
wyckoff_ge = [0.162984, 0.337016, 0.662984]
wyckoff_rh = [0.126592, 0.126592, 0.126592]
struc = Structure.from_spacegroup(sg, [[a, 0, 0], [0, a, 0], [0, 0, a]], ['Rh'], [wyckoff_rh])

# Construct a Model instance
model = TightBindingModel(struc)

model.generate_couplings(5, sg=sg)
model.show_couplings()

v1 = 0.78
vp = -0.76
v2 = 0.16
model.set_coupling(v1, 0)
model.set_coupling(vp, 1)
model.set_coupling(v2, 3)
#%%

nk = 100
hsp_labels = ['X', 'G', 'R', 'M', 'G']
ks = np.concatenate((np.linspace([0.5, 0, 0], [0, 0, 0], nk, endpoint=False),
                     np.linspace([0, 0, 0], [0.5, 0.5, 0.5], int(np.round(np.sqrt(3) * nk)), endpoint=False),
                     np.linspace([0.5, 0.5, 0.5], [0.5, 0.5, 0], nk, endpoint=False),
                     np.linspace([0.5, 0.5, 0], [0, 0, 0], int(np.round(np.sqrt(2) * nk)), endpoint=True)))

spec = Spec(model, ks)

fig, ax = plt.subplots()
for band in spec.E.T:
    ax.plot(np.arange(len(ks)), band, c='k')
plt.show()

#%%