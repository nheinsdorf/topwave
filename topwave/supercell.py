from itertools import product

import numpy as np
import numpy.typing as npt
from pymatgen.core.structure import Lattice, Structure
from topwave.coupling import Coupling
from topwave.model import Model, SpinWaveModel, TightBindingModel
from topwave import util
from topwave.types import RealList, Vector

#__all__ = ["Supercell"]
#NOTE: what about zeeman term? Just warning that it has been put to zero?
def get_supercell(model: Model,
                  scaling_factors: Vector) -> Model:
    """Creates a supercell of a given model.

    """

    supercell_structure = _get_supercell_structure(model, scaling_factors)
    supercell_model = object.__new__(type(model), supercell_structure)
    supercell_model.__init__(supercell_structure, import_site_properties=True)

    supercell_model.scaling_factors = scaling_factors
    supercell_model._is_spin_polarized = model._is_spin_polarized

    _generate_supercell_couplings(supercell_model, model)

    return supercell_model


def _get_supercell_structure(model: Model,
                             scaling_factors: Vector) -> Structure:
    """Creates a supercell structure.

    """

    lattice = (scaling_factors * model.structure.lattice.matrix.T).T

    num_uc = np.prod(scaling_factors)
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
def _generate_supercell_couplings(supercell: Model,
                                  model: Model):
    """Copies the couplings of the model to the respective bonds of the supercell."""

    cell_vectors = np.array([site.properties['cell_vector'] for site in supercell.structure], dtype=int)
    cell_vectors = np.unique(cell_vectors, axis=0)

    num_uc = np.prod(supercell.scaling_factors)
    x_lim, y_lim, z_lim = supercell.scaling_factors
    dim = len(model.structure)
    for _, coupling in enumerate(model.couplings):
        for cell_index, (x, y, z) in enumerate(product(range(x_lim), range(y_lim), range(z_lim))):
            target_cell = np.mod(coupling.lattice_vector + [x, y, z], supercell.scaling_factors)
            target_cell_index = np.arange(num_uc)[np.all(cell_vectors == target_cell, axis=1)][0]
            # print(coupling.site1.properties['index'])
            lattice_vector = np.floor_divide(coupling.lattice_vector + [x, y, z], supercell.scaling_factors)
            site1 = supercell.structure[coupling.site1.properties['index'] + dim * cell_index]
            site2 = supercell.structure[coupling.site2.properties['index'] + dim * target_cell_index]
            new_coupling_index = _ * num_uc + cell_index
            new_coupling = Coupling(new_coupling_index, lattice_vector, site1, coupling.orbital1,
                                    site2, coupling.orbital2, coupling.symmetry_id, coupling.symmetry_op)
            supercell.couplings.append(new_coupling)
            supercell.set_coupling(new_coupling_index, coupling.strength, attribute='index')
            supercell.set_spin_orbit(new_coupling_index, coupling.spin_orbit, attribute='index')


def stack(layers: list[Model],
          spacings: RealList,
          direction: str = 'z',
          vacuum: float = 10) -> Model:
    """Creates stacks of models in a given stacking direction.

    This function creates stacked models. Onsite properties and intralayer couplings are transferred to the new model.
    Because the resulting model is usually treated as two-dimensional, the method adds vacuum to the new unit cell.

    .. admonition:: Compatible Lattices
            :class: warning

            This method assumes that the models have the same lattice structure, i.e. the unit cell area within
            the plane is not increased by the stacking.


    Parameters
    ----------
    layers: list[Model]
        A list of models that are stacked on top of each other.
    spacings: RealList
        List of floats that specify the interlayer distances in Angstrom. Length should be the number of models minus 1.
    direction: str
        A string indicating along which lattice vector the models are stacked. Options are 'x', 'y' and 'z'.
        Default is 'z'.
    vacuum: float
        How much vacuum in Angstrom is added to the stacked unit cell. Default is 10.

    Returns
    -------
    Model
        The stacked model.

    Examples
    --------

    Create an bernal bilayer of graphene.

    .. ipython:: python

        # build two graphene models with staggered displacement fields (onsite terms)
        lattice = Lattice.hexagonal(1.42, 1)
        layer_A = Structure.from_spacegroup(sg=191, lattice=lattice, species=['C'], coords=[[1 / 3, 2 / 3, 0]])
        layer_B = layer_A.copy()
        # shift layer_B to get bernal-type stacking
        layer_B.translate_sites(indices=[0, 1], vector=layer_A.frac_coords[0])
        model_A, model_B = tp.TightBindingModel(layer_A), tp.TightBindingModel(layer_B)

        displacement_field = 0.2
        model_A.set_onsite_scalar(1, displacement_field)
        model_B.set_onsite_scalar(0, -displacement_field)

        # stack the models with interlayer distance 3.44 Angstrom
        model_bernal = tp.stack([model_A, model_B], [3.44])

        # the onsite terms have been transferred (see column 'onsite scalar')
        model_bernal.show_site_properties()
        # plot the supercell when supercell is updated

    """

    structures = [layer.structure.copy() for layer in layers]
    stacked_structure = _get_stacked_structure(structures, spacings, direction, vacuum)
    stacked_model = object.__new__(type(layers[0]), stacked_structure)
    stacked_model.__init__(stacked_structure, import_site_properties=True)

    return stacked_model

def _get_stacked_structure(layers: list[Structure],
                           spacings: RealList,
                           direction: str = 'z',
                           vacuum: float = 10) -> Structure:
    """Creates a stacked structure in a given stacking direction.


    Parameters
    ----------
    layers: list[Structure]
        A list of structures that are stacked on top of each other.
    spacings: RealList
        List of floats that specify the interlayer distances in Angstrom. Length should be the number of models minus 1.
    direction: str
        A string indicating along which lattice vector the models are stacked. Options are 'x', 'y' and 'z'.
        Default is 'z'.
    vacuum: float
        How much vacuum in Angstrom is added to the stacked unit cell. Default is 10.

    Returns
    -------
    Structure
        The stacked structure.

    """


    normal_index = 'xyz'.find(direction)
    plane_indices = np.delete(np.arange(3), normal_index)
    stack_height = np.array(spacings, dtype=np.float64).sum()
    layer_positions = np.concatenate(([0], np.cumsum(spacings)), dtype=np.float64, axis=0)
    lattice_parameters = list(layers[0].lattice.parameters)
    lattice_parameters[normal_index] = stack_height + vacuum
    new_lattice = Lattice.from_parameters(*lattice_parameters)
    structure = Structure.from_spacegroup(sg=1, lattice=new_lattice, species=[], coords=[])

    for layer_index, (layer, layer_position) in enumerate(zip(layers, layer_positions)):
        for site_index, site in enumerate(layer):
            site.properties['layer'] = layer_index
            layer_shift = np.zeros(3)
            layer_shift[normal_index] = layer_position
            structure.append(site.species_string, site.coords + layer_shift,
                             properties=site.properties, coords_are_cartesian=True)

    for new_site_index, site in enumerate(structure):
        site.properties['index'] = new_site_index

    return structure


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

