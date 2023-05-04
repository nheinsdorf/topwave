#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:07:56 2022

@author: niclas
"""
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import numpy.typing as npt
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.groups import SpaceGroup
from scipy.linalg import norm
from scipy.optimize import minimize
from tabulate import tabulate

from topwave.constants import G_LANDE, MU_BOHR
from topwave.coupling import Coupling
from topwave import util

__all__ = ["Model"]
class Model(ABC):
    """Base class that contains the physical model."""

    def __init__(self,
                 structure: Structure,
                 import_site_properties: bool = False) -> None:

        self.structure = structure
        self.type = self.get_type()

        # allocate site properties and enumerate them by an index
        if not import_site_properties:
            for _, site in enumerate(self.structure):
                site.properties['index'] = _
                site.properties['magmom'] = None
                site.properties['onsite_scalar'] = 0.
                site.properties['onsite_vector'] = np.zeros(3, dtype=float)
                site.properties['orbitals'] = 1
                site.properties['Rot'] = None

                # supercell and twisted properties
                site.properties['cell_vector'] = None
                site.properties['uc_site_index'] = None
                site.properties['layer'] = None
        self.scaling_factors = None
        self.normal = None
        self.twist_tuple = None

        # put zero magnetic field
        self.zeeman = np.zeros(3, dtype=float)

        # allocate an empty list for the couplings
        self.couplings = []

    def delete_all_couplings(self) -> None:
        """Deletes all couplings."""

        self.couplings = []

    def generate_couplings(self,
                           max_distance: float,
                           space_group: int) -> None:
        """Generates couplings up to a distance and groups them based on the spacegroup."""

        neighbors = self.structure.get_symmetric_neighbor_list(max_distance, sg=space_group, unique=True)
        self.delete_all_couplings()
        index = 0
        for site1_id, site2_id, lattice_vector, _, symmetry_id, symmetry_op in zip(*neighbors):
            site1 = self.structure[site1_id]
            site2 = self.structure[site2_id]
            for orbital1, orbital2 in product(range(site1.properties['orbitals']), range(site2.properties['orbitals'])):
                coupling = Coupling(index, lattice_vector, site1, orbital1, site2, orbital2, int(symmetry_id), symmetry_op)
                self.couplings.append(coupling)
                index += 1

    def get_couplings(self,
                      attribute: str,
                      value: int | float) -> list[Coupling]:
        """Returns couplings selected by some attribute"""

        indices = util.coupling_selector(attribute=attribute, value=value, model=self)
        return [self.couplings[index] for index in indices]

    # NOTE: should I get rid of this and just replace it with get_couplings in spec?
    def get_set_couplings(self) -> list[Coupling]:
        """Returns couplings that have been assigned some exchange."""

        indices = util.coupling_selector(attribute='is_set', value=True, model=self)
        return [self.couplings[index] for index in indices]

    @abstractmethod
    def get_type(self) -> str:
        """Returns the type of the model."""

    def invert_coupling(self,
                        index: int) -> None:
        """Inverts the orientation of a coupling."""

        coupling = self.couplings[index]
        site1, site2 = coupling.site1, coupling.site2
        orbital1, orbital2 = coupling.site1.properties['orbitals'], coupling.site2.properties['orbitals']
        lattice_vector = coupling.lattice_vector
        symmetry_id, symmetry_op = coupling.symmetry_id, coupling.symmetry_op
        inverted_coupling = Coupling(index, -lattice_vector, site2, orbital2, site1, orbital1, symmetry_id, symmetry_op)
        self.couplings[index] = inverted_coupling

    def set_coupling(self,
                     attribute_value: int | float,
                     strength: float,
                     attribute: str = 'index') -> None:
        """Assigns (scalar) hopping/exchange to a selection of couplings."""

        couplings = self.get_couplings(attribute=attribute, value=attribute_value)
        for coupling in couplings:
            coupling.set_coupling(strength)

    def set_moments(self,
                    orientations: list[npt.ArrayLike],
                    magnitudes: list[float] = None) -> None:
        """Sets the magnetic moments on each site of the structure given in lattice coordinates."""

        for _, (orientation, site) in enumerate(zip(orientations, self.structure)):
            # compute or save the magnitude (in lattice coordinates!).
            magnitude = norm(orientation) if magnitudes is None else magnitudes[_]
            orientation = np.array(orientation, dtype=np.float64).reshape((3,))
            # transform into cartesian coordinates (spin frame) and normalize.
            moment = self.structure.lattice.matrix.T @ orientation
            moment = moment / norm(moment)
            # calculate rotation matrix that rotates the spin to the quantization axis.
            site.properties['Rot'] = util.rotate_vector_to_ez(moment)
            site.properties['magmom'] = moment * magnitude

    def set_onsite_scalar(self,
                          index: int,
                          strength: float,
                          space_group: int = 1) -> None:
        """Sets a scalar onsite energy to a given site."""

        space_group = SpaceGroup.from_int_number(space_group)
        coordinates = space_group.get_orbit(self.structure[index].frac_coords)
        for coordinate in coordinates:
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            site.properties['onsite_scalar'] = float(strength)

    def set_onsite_vector(self,
                          index: int, vector: list[float] | npt.NDArray[np.float64],
                          strength: float = None,
                          space_group: int = 1) -> None:
        """Sets a local Zeeman field/single-ion anisotropy to a given site."""

        input_vector = util.format_input_vector(orientation=vector, length=strength)
        space_group = SpaceGroup.from_int_number(space_group)
        coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
        for coordinate, operation in zip(coordinates, operations):
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            site.properties['onsite_vector'] = operation.apply_rotation_only(input_vector)

    def set_open_boundaries(self,
                            direction: str = 'xyz') -> None:
        """Sets the exchange/hopping and DM/SOC at the chosen boundary to zero."""

        boundary_indices = util.get_boundary_couplings(model=self, direction=direction)
        for index in boundary_indices:
            self.unset_coupling(attribute_value=index, attribute='index')

    def set_spin_orbit(self,
                       attribute_value: int | float,
                       vector: list[float] | npt.NDArray[np.float64],
                       strength: float = None,
                       attribute: str = 'index') -> None:
        """Assigns spin dependent hopping/DM exchange to a selection of couplings."""

        input_vector = util.format_input_vector(orientation=vector, length=strength)
        couplings = self.get_couplings(attribute=attribute, value=attribute_value)
        for coupling in couplings:
            spin_orbit = coupling.symmetry_op.apply_rotation_only(input_vector) if attribute == 'symmetry_id' else input_vector
            coupling.set_spin_orbit(spin_orbit)

    def set_zeeman(self,
                   orientation: list[float] | npt.NDArray[np.float64],
                   strength: float = None) -> None:
        """Sets a global Zeeman term."""

        self.zeeman = util.format_input_vector(orientation=orientation, length=strength)

    def show_couplings(self) -> None:
        """Prints the couplings."""

        header = ['index', 'symmetry index', 'symmetry operation', 'distance', 'lattice_vector', 'sublattice_vector',
                  'site1', 'orbital1', 'site2', 'orbital2', 'strength', 'spin-orbit vector']
        table = []
        for coupling in self.couplings:
            table.append([coupling.index, coupling.symmetry_id, coupling.symmetry_op.as_xyz_string(), coupling.distance,
                          coupling.lattice_vector, coupling.sublattice_vector, coupling.site1.properties['index'],
                          coupling.orbital1, coupling.site2.properties['index'], coupling.orbital2, coupling.strength,
                          coupling.spin_orbit])

        print(tabulate(table, headers=header, tablefmt='fancy_grid'))

    def show_site_properties(self) -> None:
        """Prints the site properties."""

        header = ['index', 'species', 'orbitals', 'coordinates (latt.)', 'coordinates (cart.)', 'magmom',
                  'onsite scalar', 'onsite vector', 'unit cell index', 'supercell vector', 'layer']
        table = []
        for site in self.structure:
            table.append([site.properties['index'], site.species, site.properties['orbitals'], site.frac_coords,
                          site.coords, site.properties['magmom'], site.properties['onsite_scalar'],
                          site.properties['onsite_vector'], site.properties['uc_site_index'],
                          site.properties['cell_vector'], site.properties['layer']])

        print(tabulate(table, headers=header, tablefmt='fancy_grid'))
        print(f'Zeeman: {self.zeeman}')
        print(f'Supercell Size: {self.scaling_factors}')

    def unset_coupling(self,
                       attribute_value: int | float,
                       attribute: str = 'index') -> None:
        """Removes exchanges from a coupling and makes it unset."""

        indices = util.coupling_selector(attribute=attribute, value=attribute_value, model=self)
        for _ in indices:
            self.couplings[_].unset()

    def unset_moments(self):
        """Unsets all magnetic moments of the structure."""

        for site in self.structure:
            site.properties['magmom'] = None

    def write_cif(self,
                  path: str,
                  write_magmoms: bool = True) -> None:
        """Saves the structure to a .mcif file."""

        if self.type == 'tightbinding':
            for site in self.structure:
                site.properties['magmom'] = site.properties['onsite_vector']
                CifWriter(self.structure, write_magmoms=write_magmoms).write_file(path)
                self.unset_moments()
        else:
            CifWriter(self.structure, write_magmoms=write_magmoms).write_file(path)


class SpinWaveModel(Model):
    """Child class of Model for Spinwave models."""

    # NOTE: if I can make the multiple inheritance with the abstract get-type method work, delete this again.
    def __init__(self,
                 structure: Structure,
                 import_site_properties: bool = False) -> None:
        super().__init__(structure, import_site_properties)
        self.type = 'spinwave'

    def get_classical_energy(self,
                             per_spin: bool = True) -> float:
        """Computes the classical ground state energy of the model."""

        energy = 0
        # exchange energy
        for coupling in self.couplings:
            energy += coupling.get_energy()

        # Zeeman energy and anisotropies
        for site in self.structure:
            magmom = site.properties['magmom']
            vector = site.properties['single_ion_anisotropy']
            energy += magmom @ np.diag(vector) @ magmom
            energy -= MU_BOHR * G_LANDE * (self.zeeman @ magmom)

        if per_spin:
            return energy / len(self.structure)
        return energy

    def get_type(self) -> str:
        """Overrides the 'get_type'-method to return the spinwave type."""

        return 'spinwave'

    @staticmethod
    def __get_classical_energy_wrapper(directions,
                                       magnitudes,
                                       model):
        """Private method that is used for the minimization of classical energy in 'get_classical_groundstate.

        Parameters
        ----------
        directions : list
            3N list of floats containing the components of magnetic moments in the cell which are optimized.
        moments : list
            List of N (positive) floats that give the magnitude of the magnetic moments
        """

        directions = np.array(directions, dtype=float).reshape((-1, 3))
        model.set_moments(directions, magnitudes)
        return model.get_classical_energy()

    def get_classical_groundstate(self,
                                  random_init=False):
        """Minimize the classical energy by changing the orientation of the magnetic moments."""

        moments = np.array([site.properties['magmom'] for site in self.structure], dtype=float)
        magnitudes = norm(moments, axis=1)

        # get initial moments
        x_0 = np.random.rand(self.structure.num_sites, 3) if random_init else moments

        res = minimize(SpinWaveModel.__get_classical_energy_wrapper, x_0, args=(magnitudes, self))

        # normalize the final configuration
        res.x = (res.x.reshape((-1, 3)).T / norm(res.x.reshape((-1, 3)), axis=1)).flatten(order='F')
        return res

    def set_single_ion_anisotropy(self,
                                  index: int,
                                  vector: list[float] | npt.NDArray[np.float64],
                                  strength: float = None,
                                  space_group: int = 1) -> None:
        """Assigns single-ion anisotropy to a given site (same as 'set_onsite_vector'-method)."""

        self.set_onsite_vector(index=index, vector=vector, strength=strength, space_group=space_group)


class TightBindingModel(Model):
    """Child class of Model for Tight-Binding models."""

    # NOTE: if I can make the multiple inheritance with the abstract get-type method work, delete this again.
    def __init__(self,
                 structure: Structure,
                 import_site_properties: bool = False) -> None:
        super().__init__(structure, import_site_properties)
        self.type = 'tightbinding'

    def check_if_spinful(self):
        """Checks whether the model is spinful or spinless (polarized)."""

        couplings = self.get_set_couplings()
        has_spin_orbit = any(any(coupling.spin_orbit != np.zeros(3, dtype=float)) for coupling in couplings)
        has_onsite_vector = any(any(site.properties['onsite_vector'] != np.zeros(3, dtype=float)) for site in self.structure)
        has_zeeman = any(self.zeeman != np.zeros(3, dtype=float))
        return any([has_spin_orbit, has_onsite_vector, has_zeeman])

    def get_type(self) -> str:
        """Overrides the 'get_type'-method to return the tightbinding type."""

        return 'tightbinding'

    def set_orbitals(self,
                     index: int,
                     num_orbitals: int):
        """Sets the number of orbitals on a given site."""

        self.structure[index].properties['orbitals'] = num_orbitals
