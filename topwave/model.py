#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:07:56 2022

@author: niclas
"""
from abc import ABC
from itertools import product
import logging

import numpy as np
import numpy.typing as npt
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.groups import SpaceGroup
from scipy.linalg import norm
from scipy.optimize import minimize

from topwave.constants import G_LANDE, MU_BOHR
from topwave.coupling import Coupling
from topwave import logging_messages
from topwave import util

# TODO:
# - make set_single_ion anisotropy and set_onsite the same thing
# - the two methods above can have different names in the SW- and TB model which just call the universal one
# - get rid of self.dim!!!
# - move make supercell to its own module (can be static) (together with twist)

class Model(ABC):
    """Base class that contains the physical model."""

    def __init__(self, struc: Structure) -> None:

        # save the input structure
        self.structure = struc
        # and label the sites within struc by an index
        for _, site in enumerate(self.structure):
            site.properties['index'] = _
            site.properties['magmom'] = None
            site.properties['onsite_scalar'] = 0.
            site.properties['onsite_vector'] = np.zeros(3, dtype=float)
            # site.properties['onsite_spin_matrix'] = np.eye(2)
            site.properties['single_ion_anisotropy'] = np.zeros(3, dtype=float)

        # count the number of magnetic sites and save
        self.dim = len(self.structure)

        # allocate an empty list for the couplings and a dataframe (for easy access and printing)
        self.delete_all_couplings()

        # put zero magnetic field
        self.zeeman = np.zeros(3, dtype=float)

        self.supercell = None

    def delete_all_couplings(self) -> None:
        """Deletes all couplings."""

        self.couplings = []

    def generate_couplings(self, max_distance: float, space_group: int) -> None:
        """Generates couplings up to a distance and groups them based on the spacegroup."""

        if self.supercell is not None:
            logging.warning('coupling.Couplings must be generated before the supercell.')

        neighbors = self.structure.get_symmetric_neighbor_list(max_distance, sg=space_group, unique=True)

        # save the generated couplings (overriding any old ones)
        self.delete_all_couplings()
        for index, (site1_id, site2_id, lattice_vector, _, symmetry_id, symmetry_op) in enumerate(zip(*neighbors)):
            site1 = self.structure[site1_id]
            site2 = self.structure[site2_id]
            coupling = Coupling(index, lattice_vector, site1, site2, symmetry_id, symmetry_op)
            self.couplings.append(coupling)

    def get_set_couplings(self) -> list[Coupling]:
        """Returns couplings that have been assigned some exchange."""

        indices = util.coupling_selector(attribute='is_set', value=True, model=self)
        return [self.couplings[index] for index in indices]

    def invert_coupling(self, index: int) -> None:
        """Inverts the orientation of a coupling."""

        coupling = self.couplings[index]
        site1, site2 = coupling.site1, coupling.site2
        lattice_vector = coupling.lattice_vector
        symmetry_id, symmetry_op = coupling.symmetry_id, coupling.symmetry_op
        inverted_coupling = Coupling(index, -lattice_vector, site2, site1, symmetry_id, symmetry_op)
        self.couplings[index] = inverted_coupling

    def make_supercell(self, scaling_factors: list[int] | npt.NDArray[np.int64]) -> None:
        """Constructs a supercell and generates the new couplings."""

        if len(self.couplings) == 0:
            logging.warning('coupling.Couplings must be generated before the supercell.')

        logging.debug('Unit cell is enlarged to supercell.')
        lattice = (scaling_factors * self.structure.lattice.matrix.T).T

        num_uc = np.product(scaling_factors)
        x_lim, y_lim, z_lim = scaling_factors
        coords = []
        cell_vectors = []
        dim = len(self.structure)
        for site in self.structure:
            for (x, y, z) in product(range(x_lim), range(y_lim), range(z_lim)):
                coords.append((site.frac_coords + [x, y, z]) / scaling_factors)
                cell_vectors.append([x, y, z])
        coords = np.array(coords, dtype=float).reshape((num_uc, dim, 3), order='F')
        coords = coords.reshape((num_uc * dim), 3)
        cell_vectors = np.array(cell_vectors, dtype=int).reshape((num_uc, dim, 3), order='F')
        cell_vectors = cell_vectors.reshape((num_uc * dim), 3)

        species = [site.species_string for site in self.structure] * num_uc
        supercell = Structure.from_spacegroup(1, lattice, species, coords)
        supercell.scaling_factors = scaling_factors
        logging.debug(f'{x_lim}x{y_lim}x{z_lim} supercell has been created. Site properties are transferred.')

        for site_index, site in enumerate(self.structure):
            for _ in range(site_index, dim * num_uc, dim):
                for key, value in site.properties.items():
                    supercell[_].properties[key] = value
                supercell[_].properties['index'] = _
                supercell[_].properties['uc_site_index'] = site_index
                supercell[_].properties['cell_vector'] = cell_vectors[_]

        self.supercell = supercell
        logging.debug('Site properties have been transferred. coupling.Couplings for supercell are created.')

        uc_couplings = self.couplings.copy()
        self.delete_all_couplings()
        cell_vectors = np.unique(cell_vectors, axis=0)
        for _, coupling in enumerate(uc_couplings):
            for cell_index, (x, y, z) in enumerate(product(range(x_lim), range(y_lim), range(z_lim))):
                target_cell = np.mod(coupling.lattice_vector + [x, y, z], scaling_factors)
                target_cell_index = np.arange(num_uc)[np.all(cell_vectors == target_cell, axis=1)][0]
                R = np.floor_divide(coupling.lattice_vector + [x, y, z], scaling_factors)
                site1 = self.supercell[coupling.site1.properties['index'] + dim * cell_index]
                site2 = self.supercell[coupling.site2.properties['index'] + dim * target_cell_index]
                new_coupling_index = _ * num_uc + cell_index
                new_coupling = Coupling(new_coupling_index, R, site1, site2, coupling.symmetry_id, coupling.symmetry_op)
                self.couplings.append(new_coupling)
                self.set_coupling(new_coupling_index, coupling.strength, attribute='index')
                self.set_spin_orbit(new_coupling_index, coupling.spin_orbit, attribute='index')

        logging.debug('coupling.Couplings for supercell have been created.')

    def set_coupling(self, attribute_value: int | float, strength: float, attribute: str = 'index') -> None:
        """Assigns (scalar) hopping/exchange to a selection of couplings."""

        indices = util.coupling_selector(attribute=attribute, value=attribute_value, model=self)
        for _ in indices:
            self.couplings[_].strength = strength
            self.couplings[_].is_set = True

    def unset_coupling(self, attribute_value: int | float, attribute: str = 'index') -> None:
        """Removes exchanges from a coupling and makes it unset."""

        indices = util.coupling_selector(attribute=attribute, value=attribute_value, model=self)
        for _ in indices:
            self.couplings[_].strength = 0.
            self.couplings[_].spin_orbit = np.zeros(3, dtype=np.float64)
            self.couplings[_].is_set = False

    def unset_moments(self):
        """Unsets all magnetic moments of the structure."""

        for site in self.structure:
            site.properties['magmom'] = None

    def set_spin_orbit(self, attribute_value: int | float, vector: list[float] | npt.NDArray[np.float64], strength: float = None, attribute: str = 'index') -> None:
        """Assigns spin dependent hopping/DM exchange to a selection of couplings."""

        input_vector = util.format_input_vector(orientation=vector, length=strength)
        indices = util.coupling_selector(attribute=attribute, value=attribute_value, model=self)
        for _ in indices:
            self.couplings[_].spin_orbit = self.couplings[_].symmetry_op.apply_rotation_only(input_vector) if attribute == 'symmetry_id' else input_vector
            self.couplings[_].is_set = True

        if isinstance(self, TightBindingModel):
            self.make_spinful()

    def set_onsite_vector(self, index: int, vector: list[float] | npt.NDArray[np.float64], strength: float = None, space_group: int = 1) -> None:
        """Sets a local Zeeman field/single-ion anisotropy to a given site."""

        input_vector = util.format_input_vector(orientation=vector, length=strength)
        if self.supercell is None:
            space_group = SpaceGroup.from_int_number(space_group)
            coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
            for coordinate, operation in zip(coordinates, operations):
                cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
                site = self.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
                site.properties['onsite_vector'] = operation.apply_rotation_only(input_vector)
        else:
            logging.warning(logging_messages.SET_SITE_PROPERTY_SUPERCELL)
            self.supercell[index].properties['onsite_vector'] = input_vector

        if isinstance(self, TightBindingModel):
            self.make_spinful()

    def set_onsite_scalar(self, index: int, strength: float, space_group: int = 1) -> None:
        """Sets a scalar onsite energy to a given site."""

        if self.supercell is None:
            space_group = SpaceGroup.from_int_number(space_group)
            coordinates = space_group.get_orbit(self.structure[index].frac_coords)
            for coordinate in coordinates:
                cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
                site = self.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
                site.properties['onsite_scalar'] = strength
        else:
            logging.warning(logging_messages.SET_SITE_PROPERTY_SUPERCELL)
            self.supercell[index].properties['onsite_scalar'] = strength

    def set_zeeman(self, orientation: list[float] | npt.NDArray[np.float64], strength: float = None) -> None:
        """Sets a global Zeeman term."""

        self.zeeman = util.format_input_vector(orientation=orientation, length=strength)

        if isinstance(self, TightBindingModel):
            self.make_spinful()

    def set_moments(self, directions, magnitudes):
        """ Assigns a magnetic ground state to the model


        Parameters
        ----------
        directions : list
            List of lists (or numpy.ndarray) that contains a three-dimensional
            spin vector for each magnetic site in the unit cell indicating the
            direction of the spin on each site (given in units of the lattice vectors).
            Each vector is normalized, so only the direction matters. The magnitude of
            magnetic moment is indicated by the 'magnitudes' argument. The vector is given in
            the units of the lattice vectors.
        magnitudes : list
            List of floats specifying the magnitude of magnetic moment of each site.


        """

        # TODO: Add option to add magnetic moments to supercell
        if self.supercell is not None:
            N_SC = len(self.supercell)
            try:
                directions = np.array(directions, dtype=float).reshape((N_SC, 3))
                magnitudes = np.array(magnitudes, dtype=float).reshape((N_SC,))
                struc = self.supercell
                #logging.warning('Magnetic moments must be set before the supercell.')
            except:
                directions = np.array(directions, dtype=float).reshape((self.dim, 3))
                magnitudes = np.array(magnitudes, dtype=float).reshape((self.dim,))
                struc = self.structure
        else:
            directions = np.array(directions, dtype=float).reshape((self.dim, 3))
            magnitudes = np.array(magnitudes, dtype=float).reshape((self.dim,))
            struc = self.structure

        for _, (direction, magnitude) in enumerate(zip(directions, magnitudes)):
            # rotate into cartesian coordinates and normalize it
            moment = struc.lattice.matrix.T @ direction
            moment = moment / norm(moment)

            # calculate the rotation matrix that rotates the spin to the quantization axis
            struc[_].properties['Rot'] = util.rotate_vector_to_ez(moment)
            # stretch it to match the right magnetic moment and save it
            struc[_].properties['magmom'] = moment * magnitude

    def set_open_boundaries(self, direction: str = 'xyz') -> None:
        """Sets the exchange/hopping and DM/SOC at the chosen boundary to zero."""

        boundary_indices = util.get_boundary_couplings(model=self, direction=direction)
        for index in boundary_indices:
            self.unset_coupling(attribute_value=index, attribute='index')

    def write_cif(self, path: str, write_magmoms: bool = True) -> None:
        """Saves the structure to a .mcif file."""

        if isinstance(self, TightBindingModel) and write_magmoms:
            logging.info(logging_messages.TIGHTBINDING_MCIF)
            for site in self.structure:
                site.properties['magmom'] = site.properties['onsite_vector']
                CifWriter(self.structure, write_magmoms=write_magmoms).write_file(path)
                self.unset_moments()
        else:
            CifWriter(self.structure, write_magmoms=write_magmoms).write_file(path)


class SpinWaveModel(Model):
    """Child class of Model for Spinwave models."""

    def get_classical_energy(self, per_spin: bool = True) -> float:
        """Computes the classical ground state energy of the model.

        Parameters
        ----------
        per_spin : bool
            If true, the total energy is divided by the number of magnetic sites in the model.
            Default is true.

        Returns
        -------
        energy : float
            Total classical ground state energy or energy per spin if per_spin is True.

        """

        energy = 0
        # exchange energy
        for coupling in self.couplings:
            energy += coupling.get_energy()

        struc = self.structure if self.supercell is None else self.supercell

        # Zeeman energy and anisotropies
        for site in struc:
            magmom = site.properties['magmom']
            K = site.properties['single_ion_anisotropy']
            energy += magmom @ np.diag(K) @ magmom
            energy -= MU_BOHR * G_LANDE * (self.zeeman @ magmom)

        if per_spin:
            return energy / len(struc)
        else:
            return energy

    @staticmethod
    def __get_classical_energy_wrapper(directions, magnitudes, model):
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

    def get_classical_groundstate(self, random_init=False):
        """Tries to find the classical ground state by minimizing the classical groundstate
        energy w.r.t. to the orientation of magnetic moments. Their magnitude is not considered.
        The magnetic moments of the model are set in-place.

        Parameters
        ----------
        random_init : bool
            If true, the initial direction of the moments will be random. If False, the set moments of the
            model are used. Default is False.

        Returns
        -------
        res : scipy.optimize.OptimizeResult
            Result of the minimization.
        """

        struc = self.structure if self.supercell is None else self.supercell

        moments = np.array([site.properties['magmom'] for site in struc], dtype=float)
        magnitudes = norm(moments, axis=1)

        # get initial moments
        x_0 = np.random.rand(struc.num_sites, 3) if random_init else moments

        res = minimize(SpinWaveModel.__get_classical_energy_wrapper, x_0, args=(magnitudes, self))

        # normalize the final configuration
        res.x = (res.x.reshape((-1, 3)).T / norm(res.x.reshape((-1, 3)), axis=1)).flatten(order='F')
        return res

    def set_single_ion_anisotropy(self, index: int, vector: list[float] | npt.NDArray[np.float64], strength: float = None, space_group: int = 1) -> None:
        """Assigns single-ion anisotropy to a given site."""

        self.set_onsite_vector(index=index, vector=vector, strength=strength, space_group=space_group)

class TightBindingModel(Model):
    """Child class of Model for Tight-Binding models."""

    def __init__(self, struc):
        super().__init__(struc)
        self.spinful = False

    def make_spinful(self):
        """Adds spin as a degree of freedom to the model."""

        self.spinful = True
