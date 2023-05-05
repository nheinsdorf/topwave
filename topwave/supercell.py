from itertools import product

import numpy as np
import numpy.typing as npt
from pymatgen.core.structure import Structure
from topwave.coupling import Coupling
from topwave.model import Model, SpinWaveModel, TightBindingModel
from topwave import util

__all__ = ["Supercell"]

class Supercell(SpinWaveModel, TightBindingModel):
    """Supercell of a given model."""

    # NOTE: Check the tuple input and convert to list (or everything to tuple). But make sure type is always the same.
    def __init__(self, model: Model, scaling_factors: tuple[int, int, int] | list[int] | npt.NDArray[np.int64]) -> None:

        super_structure = Supercell.get_supercell_structure(model, scaling_factors)

        if isinstance(model, SpinWaveModel):
            SpinWaveModel.__init__(self, super_structure, import_site_properties=True)
        else:
            TightBindingModel.__init__(self, super_structure, import_site_properties=True)

        self.scaling_factors = scaling_factors
        self._generate_supercell_couplings(model)
        self.zeeman = model.zeeman

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
                new_coupling = Coupling(new_coupling_index, lattice_vector, site1, coupling.orbital1,
                                        site2, coupling.orbital2, coupling.symmetry_id, coupling.symmetry_op)
                self.couplings.append(new_coupling)
                self.set_coupling(new_coupling_index, coupling.strength, attribute='index')
                self.set_spin_orbit(new_coupling_index, coupling.spin_orbit, attribute='index')
#
# class Twist(SpinWaveModel, TightBindingModel):
#     """Adds a twisted layer to a given model."""
#
#     def __init__(self, model: Model, normal: str, twist_tuple: tuple[int, int] | list[int] | npt.NDArray[np.int64]) -> None:
#
#         super_structure = Supercell.get_supercell_structure(model, normal, twist_tuple)
#
#         if isinstance(model, SpinWaveModel):
#             SpinWaveModel.__init__(self, super_structure, import_site_properties=True)
#         else:
#             TightBindingModel.__init__(self, super_structure, import_site_properties=True)
#
#         self.normal = normal
#         self.twist_tuple = twist_tuple
#         # self._generate_supercell_couplings(model)
#         self.zeeman = model.zeeman
#
#     @staticmethod
#     def get_supercell_structure(model: Model, normal: str, twist_tuple: tuple[int, int] | list[int] | npt.NDArray[np.int64]) -> Structure:
#         """Constructs a twisted layer"""
#
#         id_x, id_y, id_z = util.get_span_indices(normal)
#
#         # construct the lattice of the supercell
#         lattice_vector1 = twist_tuple[0] * model.structure.lattice.matrix[id_x] \
#                           + twist_tuple[1] * model.structure.lattice.matrix[id_y]
#         lattice_vector2 = twist_tuple[0] * model.structure.lattice.matrix[id_x] \
#                           - twist_tuple[1] * model.structure.lattice.matrix[id_y]
#         lattice_vector3 = model.structure.lattice.matrix[id_z]
#         lattice = np.array([lattice_vector1, lattice_vector2, lattice_vector3])
#
#         twist_angle = util.get_angle(lattice_vector1, lattice_vector2)
#         rotation_axis = np.zeros(3, dtype=np.float64)
#         rotation_axis[id_z] = 1.0
#
#         scaling_factors = np.zeros(3, dtype=np.int)
#         scaling_factors[id_x] = twist_tuple[0]
#         scaling_factors[id_y] = twist_tuple[1]
#         lower_layer = Supercell(model, scaling_factors)
#
#         coords = []
#         for site in lower_layer:
#             util.rotate_vector(site.coords, twist_angle
#
#
#
#
#         num_uc = np.product(scaling_factors)
#         x_lim, y_lim, z_lim = scaling_factors
#         coords = []
#         cell_vectors = []
#         dim = len(model.structure)
#         for site in model.structure:
#             for (x, y, z) in product(range(x_lim), range(y_lim), range(z_lim)):
#                 coords.append((site.frac_coords + [x, y, z]) / scaling_factors)
#                 cell_vectors.append([x, y, z])
#         coords = np.array(coords, dtype=float).reshape((num_uc, dim, 3), order='F')
#         coords = coords.reshape((num_uc * dim), 3)
#         cell_vectors = np.array(cell_vectors, dtype=int).reshape((num_uc, dim, 3), order='F')
#         cell_vectors = cell_vectors.reshape((num_uc * dim), 3)
#         species = [site.species_string for site in model.structure] * num_uc
#
#         for site_index, site in enumerate(model.structure):
#             for _ in range(site_index, dim * num_uc, dim):
#                 for key, value in site.properties.items():
#                     supercell[_].properties[key] = value
#                 supercell[_].properties['index'] = _
#                 supercell[_].properties['cell_vector'] = cell_vectors[_]
#                 supercell[_].properties['uc_site_index'] = site_index
#
#         return supercell

