from itertools import product

import numpy as np
import numpy.typing as npt
from pymatgen.core.structure import Structure
from topwave.coupling import Coupling
from topwave.model import Model, SpinWaveModel, TightBindingModel


class Supercell(SpinWaveModel, TightBindingModel):
    """Supercell of a given model."""

    def __init__(self, model: Model, scaling_factors: list[int] | npt.NDArray[np.int64]) -> None:

        super_structure = Supercell.get_supercell_structure(model, scaling_factors)
        if isinstance(model, SpinWaveModel):
            SpinWaveModel.__init__(self, super_structure)
        else:
            TightBindingModel.__init__(self, super_structure)

        self.scaling_factors = scaling_factors
        self._generate_supercell_couplings(model)

    @staticmethod
    def get_supercell_structure(model: Model, scaling_factors: list[int] | npt.NDArray[np.int64]) -> Structure:
        """Constructs a supercell out of an existing model."""

        lattice = (scaling_factors * model.structure.lattice.matrix.T).T

        num_uc = np.product(scaling_factors)
        x_lim, y_lim, z_lim = scaling_factors
        coords = []
        cell_vectors = []
        dim = len(model.structure)
        for site in model.structure:
            for (x, y, z) in product(range(x_lim), range(y_lim), range(z_lim)):
                coords.append((site.frac_coords + [x, y, z]) / scaling_factors)
                cell_vectors.append([x, y, z])
        coords = np.array(coords, dtype=float).reshape((num_uc, dim, 3), order='F')
        coords = coords.reshape((num_uc * dim), 3)
        cell_vectors = np.array(cell_vectors, dtype=int).reshape((num_uc, dim, 3), order='F')
        cell_vectors = cell_vectors.reshape((num_uc * dim), 3)
        species = [site.species_string for site in model.structure] * num_uc
        supercell = Structure.from_spacegroup(1, lattice, species, coords)

        for site_index, site in enumerate(model.structure):
            for _ in range(site_index, dim * num_uc, dim):
                for key, value in site.properties.items():
                    supercell[_].properties[key] = value
                supercell[_].properties['index'] = _
                supercell[_].properties['cell_vector'] = cell_vectors[_]
                supercell[_].properties['uc_site_index'] = site_index

        return supercell

    def _generate_supercell_couplings(self, model: Model):
        """Copies the couplings of the model to the respective bonds of the supercell."""

        cell_vectors = np.array([site.properties['cell_vector'] for site in self.structure], dtype=int)
        cell_vectors = np.unique(cell_vectors, axis=0)

        num_uc = np.product(self.scaling_factors)
        x_lim, y_lim, z_lim = self.scaling_factors
        dim = len(model.structure)
        for _, coupling in enumerate(model.couplings):
            for cell_index, (x, y, z) in enumerate(product(range(x_lim), range(y_lim), range(z_lim))):
                target_cell = np.mod(coupling.lattice_vector + [x, y, z], self.scaling_factors)
                target_cell_index = np.arange(num_uc)[np.all(cell_vectors == target_cell, axis=1)][0]
                # print(coupling.site1.properties['index'])
                lattice_vector = np.floor_divide(coupling.lattice_vector + [x, y, z], self.scaling_factors)
                site1 = self.structure[coupling.site1.properties['index'] + dim * cell_index]
                site2 = self.structure[coupling.site2.properties['index'] + dim * target_cell_index]
                new_coupling_index = _ * num_uc + cell_index
                new_coupling = Coupling(new_coupling_index, lattice_vector, site1, site2, coupling.symmetry_id,
                                        coupling.symmetry_op)
                self.couplings.append(new_coupling)
                self.set_coupling(new_coupling_index, coupling.strength, attribute='index')
                self.set_spin_orbit(new_coupling_index, coupling.spin_orbit, attribute='index')
