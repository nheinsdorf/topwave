#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:07:56 2022

@author: niclas
"""
from itertools import product
import logging
from pathlib import Path

import numpy as np
from numpy.linalg import eig, eigh, eigvals, multi_dot
import pandas as pd
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.groups import SpaceGroup
from scipy.linalg import block_diag, norm
from scipy.optimize import minimize
import sympy as sp
from tabulate import tabulate

from topwave.coupling import Coupling
from topwave.util import rotate_vector_to_ez

# TODO:
# - make set_DM and set_SOC the same thing. Something like set_vector_exchange or something
# - make set_single_ion anisotropy and set_onsite the same thing
# - the two methods above can have different names in the SW- and TB model which just call the universal one
# - always work with the supercell and make it a [1, 1, 1] supercell by default

class ModelMixin:
    """Base class that contains the physical model.

    Parameters
    ----------
    struc : pymatgen.core.Structure
        pymatgen Structure that contains all the sites of the model.

    Attributes
    ----------
    STRUC : pymatgen.core.Structure
        This is where struc is stored
    N : int
        Number of sites in STRUC
    CPLS : list
        This is where all the couplings of the model are stored.
    CPLS_df : pandas.core.frame.DataFrame
        Attributes of CPLS are stored here for printing and selecting.
    MF : numpy.ndarray
        This is variable holds the external magnetic field. It is set via
        'set_field'. Default is [0, 0, 0].
    supercell : pymatgen.core.Structure
        The structure that holds the supercell constructed by the 'make_supercell'-method.


    Methods
    -------
    generate_couplings(maxdist, sg, supercell):
        Given a maximal distance (in Angstrom) all periodic bonds are
        generated and grouped by symmetry based on the provided sg.
    get_boundary_couplings(direction):
        Returns a list of couplings that couple sites in adjacent unit cells in the direction
        given by the provided string. E.g.: 'x' returns all couplings that couple sites in the
        adjacent unit cell in the direction of the first lattice vector, whereas 'xyz' returns those
        that couple sites from any other unit cell. Default is 'xyz'.
        Returns a list of couplings that
    invert_coupling(index):
        Invert the order of a given coupling.
    make_supercell(scaling_factors):
        Constructs a supercell.
    remove_coupling(index, by_symmetry):
        Removes a coupling.
    show_couplings():
        Prints the couplings.
    set_coupling(strength, index, by_symmetry, label):
        Assign Heisenberg Exchange or hopping amplitude terms to a collection of couplings based
        on their symmetry index.
    set_field(B):
        Apply an external magnetic field that is stored in self.MF
    set_moments(gs):
        Provide a classical magnetic ground state.
    set_open_boundaries(direction):
        Sets the strength of exchange/hopping and of DM/SOC for all couplings that couple sites
        to adjacent unit cells in the given direction to zero. Default is for all directions.
    show_moments():
        Prints the magnetic moments.
    show_anisotropies():
        Prints the single ion anisotropies.
    write_cif():
        A function that uses pymatgen functionality to save ModelMixin.STRUC
        as a mcif file to be visualized in e.g. VESTA

    """

    # maybe i should save these in topwave itself
    muB = 0.057883818066000  # given in meV/Tesla
    g = 2  # for now let's fix the g-tensor to 2 (as a number)
    kB = 0.086173324 # given in meV/K

    def __init__(self, struc):

        # save the input structure
        self.STRUC = struc
        # and label the sites within struc by an index
        for _, site in enumerate(self.STRUC):
            site.properties['id'] = _
            site.properties['magmom'] = None
            site.properties['onsite_label'] = None
            site.properties['onsite_strength'] = 0
            site.properties['onsite_spin_matrix'] = np.eye(2)
            site.properties['single_ion_anisotropy'] = np.zeros(3, dtype=float)

        # count the number of magnetic sites and save
        self.N = len(self.STRUC)

        # allocate an empty list for the couplings and a dataframe (for easy access and printing)
        self.CPLS = None
        self.CPLS_as_df = None
        self.reset_all_couplings()

        # put zero magnetic field
        self.MF = np.zeros(3, dtype=float)

        self.supercell = None

    def reset_all_couplings(self):
        """
        Deletes all couplings from self.CPLS and self.CPLS_as_df and allocates the
        name space to an empty list/df.

        """
        self.CPLS = []
        self.CPLS_as_df = pd.DataFrame(columns=['symid', 'symop', 'delta', 'R', 'dist', 'i',
                                                'at1', 'j', 'at2', 'strength', 'DM'])

    def generate_couplings(self, maxdist, sg):
        """
        Generates couplings up to a distance maxdist and stores them


        Parameters
        ----------
        maxdist : float
            Distance (in Angstrom) up to which couplings are generated.
        sg : int, optional
            International Space Group number.


        """

        if self.supercell is not None:
            logging.warning('Couplings must be generated before the supercell.')

        cpls = self.STRUC.get_symmetric_neighbor_list(maxdist, sg=sg, unique=True)

        # save the generated couplings (overriding any old ones)
        self.reset_all_couplings()
        for cplid, (i, j, R, d, symid, symop) in enumerate(zip(*cpls)):
            site1 = self.STRUC[i]
            site2 = self.STRUC[j]
            cpl = Coupling(site1, site2, cplid, symid, symop, R)
            self.CPLS.append(cpl)
            self.CPLS_as_df = pd.concat([self.CPLS_as_df, cpl.DF])
        self.CPLS_as_df.reset_index(drop=True, inplace=True)

    def get_boundary_couplings(self, direction='xyz'):
        """Returns a list of couplings that couple sites in adjacent unit cells in the a given direction.

        Parameters
        ----------
        direction : str
            Specifies in which direction the boundary is set. E.g.: 'x' returns all couplings that couple
            sites in the adjacent unit cell in the direction of the first lattice vector, whereas 'xyz' returns those
            that couple sites from any other unit cell. Default is 'xyz'.

        Returns
        -------
        tuple[list, list]
            List of topwave.coupling and the their indices w.r.t. self.CPLS.

        """

        Rs = np.array([cpl.R for cpl in self.CPLS], dtype=float)

        if len(Rs) == 0:
            boundary_couplings = boundary_indices = []
        else:
            x_indices = y_indices = z_indices = np.array([], dtype=int).reshape((0,))
            if 'x' in direction:
                x_indices = np.arange(len(self.CPLS))[Rs[:, 0] != 0]
            if 'y' in direction:
                y_indices = np.arange(len(self.CPLS))[Rs[:, 1] != 0]
            if 'z' in direction:
                z_indices = np.arange(len(self.CPLS))[Rs[:, 2] != 0]
            boundary_indices = np.unique(np.concatenate((x_indices, y_indices, z_indices), axis=0))
            boundary_couplings = [self.CPLS[_] for _ in boundary_indices]

        return boundary_couplings, boundary_indices

    def invert_coupling(self, index):
        """Inverts the order of a coupling.
        """

        cpl = self.CPLS[index]
        site1, site2, R, symid, symop = cpl.SITE1, cpl.SITE2, cpl.R, cpl.SYMID, cpl.SYMOP
        inverted_coupling = Coupling(site2, site1, index, symid, symop, -R)
        self.CPLS[index] = inverted_coupling
        self.CPLS_as_df = pd.DataFrame(columns=['symid', 'symop', 'delta', 'R', 'dist', 'i',
                                                'at1', 'j', 'at2', 'strength', 'DM'])
        for cpl in self.CPLS:
            self.CPLS_as_df = pd.concat([self.CPLS_as_df, cpl.DF])
        self.CPLS_as_df.reset_index(drop=True, inplace=True)

    def make_supercell(self, scaling_factors):
        """Constructs a supercell and the corresponding couplings.

        Parameters
        ----------
        scaling_factors : list
            List of three integers that are used to scale the existing primitive unit cell.
            E.g. [2, 1 ,1] corresponds to a supercell of dimensions (2a, b, c).

        """

        if len(self.CPLS) == 0:
            logging.warning('Couplings must be generated before the supercell.')

        logging.debug('Unit cell is enlarged to supercell.')
        lattice = (scaling_factors * self.STRUC.lattice.matrix.T).T

        num_uc = np.product(scaling_factors)
        x_lim, y_lim, z_lim = scaling_factors
        coords = []
        cell_vectors = []
        for site in self.STRUC:
            for (x, y, z) in product(range(x_lim), range(y_lim), range(z_lim)):
                coords.append((site.frac_coords + [x, y, z]) / scaling_factors)
                cell_vectors.append([x, y, z])
        coords = np.array(coords, dtype=float).reshape((num_uc, self.N, 3), order='F')
        coords = coords.reshape((num_uc * self.N), 3)
        cell_vectors = np.array(cell_vectors, dtype=int).reshape((num_uc, self.N, 3), order='F')
        cell_vectors = cell_vectors.reshape((num_uc * self.N), 3)

        species = [site.species_string for site in self.STRUC] * num_uc
        supercell = Structure.from_spacegroup(1, lattice, species, coords)
        supercell.scaling_factors = scaling_factors
        logging.debug(f'{x_lim}x{y_lim}x{z_lim} supercell has been created. Site properties are transferred.')

        for site_index, site in enumerate(self.STRUC):
            for _ in range(site_index, self.N * num_uc, self.N):
                for key, value in site.properties.items():
                    supercell[_].properties[key] = value
                supercell[_].properties['id'] = _
                supercell[_].properties['uc_site_index'] = site_index
                supercell[_].properties['cell_vector'] = cell_vectors[_]

        self.supercell = supercell
        logging.debug('Site properties have been transferred. Couplings for supercell are created.')

        uc_couplings = self.CPLS.copy()
        self.reset_all_couplings()
        cell_vectors = np.unique(cell_vectors, axis=0)
        for _, coupling in enumerate(uc_couplings):
            for cell_index, (x, y, z) in enumerate(product(range(x_lim), range(y_lim), range(z_lim))):
                target_cell = np.mod(coupling.R + [x, y, z], scaling_factors)
                target_cell_index = np.arange(num_uc)[np.all(cell_vectors == target_cell, axis=1)][0]
                R = np.floor_divide(coupling.R + [x, y, z], scaling_factors)
                site1 = self.supercell[coupling.I + self.N * cell_index]
                site2 = self.supercell[coupling.J + self.N * target_cell_index]
                coupling_index = _ * num_uc + cell_index
                cpl = Coupling(site1, site2, coupling_index, coupling.SYMID, coupling.SYMOP, R)
                self.CPLS.append(cpl)
                self.CPLS_as_df = pd.concat([self.CPLS_as_df, cpl.DF])
                self.CPLS_as_df.reset_index(drop=True, inplace=True)
                self.set_coupling(coupling.strength, coupling_index, by_symmetry=False)
                if isinstance(self, TightBindingModel):
                    self.set_spin_orbit(norm(coupling.DM), coupling.DM, coupling_index, by_symmetry=False)
                else:
                    # TODO: use the set_DM method here instead
                    # self.CPLS[coupling_index].DM = coupling.DM
                    self.set_DM(coupling.DM, coupling_index, by_symmetry=False)
        self.CPLS_as_df.reset_index(drop=True, inplace=True)
        logging.debug('Couplings for supercell have been created.')

    def remove_coupling(self, index, by_symmetry=True):
        """Removes a coupling.

        Parameters
        ----------
        index : int
            Integer that corresponds to the symmetry index of a selection of
            couplings, or to the index if by_symmetry = False.
        by_symmetry : bool
            If true, index corresponds to the symmetry index of a selection of couplings.
            If false, it corresponds to the index.

        """

        if by_symmetry:
            indices = self.CPLS_as_df.index[self.CPLS_as_df['symid'] == index].tolist()
        else:
            indices = self.CPLS_as_df.index[self.CPLS_as_df.index == index].tolist()

        new_CPLS = self.CPLS.copy()
        for _ in sorted(indices, reverse=True):
            new_CPLS.pop(_)

        self.reset_all_couplings()
        for cplid, cpl in enumerate(new_CPLS):
            site1 = self.STRUC[cpl.I]
            site2 = self.STRUC[cpl.J]
            new_cpl = Coupling(site1, site2, cplid, cpl.SYMID, cpl.SYMOP, cpl.R)
            self.CPLS.append(new_cpl)
            self.CPLS_as_df = pd.concat([self.CPLS_as_df, new_cpl.DF])
        self.CPLS_as_df.reset_index(drop=True, inplace=True)

    def show_couplings(self):
        """
        Prints the couplings

        """

        print(tabulate(self.CPLS_as_df, headers='keys', tablefmt='github',
                       showindex=True))

    def set_coupling(self, strength, index, by_symmetry=True, by_distance=False, label=None):
        """
        Assigns Heisenberg interaction or hopping amplitude to a selection
        of couplings based on their (symmetry) index.


        Parameters
        ----------
        strength : float
            Strength of the Exchange/Hopping.
        index : int
            Integer that corresponds to the symmetry index of a selection of
            couplings, or to the index if by_symmetry = False.
        by_symmetry : bool
            If true, index corresponds to the symmetry index of a selection of couplings.
            If false, it corresponds to the index.
        by_distance : bool
            If true, index corresponds to the index in list of unique coupling distances. Overrides
            by_symmetry. Default is False.
        label : str
            Label for the exchange/hopping parameter that is used for the symbolic
            representation of the Hamiltonian. If None, a label is generated based
            on the index.

        """

        tol = 5
        if by_distance:
            distances = np.unique([np.round(coupling.D, tol) for coupling in self.CPLS])
            dist = distances[index]
            indices = self.CPLS_as_df.index[np.round(self.CPLS_as_df['dist'], tol) == dist].tolist()
        elif by_symmetry:
            indices = self.CPLS_as_df.index[self.CPLS_as_df['symid'] == index].tolist()
        else:
            indices = self.CPLS_as_df.index[self.CPLS_as_df.index == index].tolist()

        for _ in indices:
            self.CPLS[_].strength = strength
            self.CPLS_as_df.loc[_, 'strength'] = strength
            self.CPLS[_].get_label(label, by_symmetry)

    def set_field(self, direction, magnitude):
        """
        Setter for self.MF an external magnetic field to the model. This will
        be translated to a Zeeman term mu_B B dot S. If the model is an instance of TightBindingModel,
        this method will call its 'make_spinful'-method.

        Parameters
        ----------
        direction : list
            Three-dimensional vector that gives direction of an external magnetic field.
        magnitude : float
            Strength of the external magnetic field.

        """

        field = np.real(magnitude) * np.array(direction, dtype=float) / norm(direction)
        self.MF[0], self.MF[1], self.MF[2] = tuple(field.tolist())

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
                directions = np.array(directions, dtype=float).reshape((self.N, 3))
                magnitudes = np.array(magnitudes, dtype=float).reshape((self.N,))
                struc = self.STRUC
        else:
            directions = np.array(directions, dtype=float).reshape((self.N, 3))
            magnitudes = np.array(magnitudes, dtype=float).reshape((self.N,))
            struc = self.STRUC

        for _, (direction, magnitude) in enumerate(zip(directions, magnitudes)):
            # rotate into cartesian coordinates and normalize it
            moment = struc.lattice.matrix.T @ direction
            moment = moment / norm(moment)

            # calculate the rotation matrix that rotates the spin to the quantization axis
            struc[_].properties['Rot'] = rotate_vector_to_ez(moment)
            # stretch it to match the right magnetic moment and save it
            struc[_].properties['magmom'] = moment * magnitude

        # NOTE: in case struc is the supercell do we need to copy over the magmoms to STRUC?

        # extract the u- and v-vectors from the rotation matrix
        for cpl in self.CPLS:
            cpl.get_uv()

    def set_open_boundaries(self, direction='xyz'):
        """Sets the exchange/hopping and DM/SOC along the boundary couplings to zero.

        Parameters
        ----------
        direction : str
            Direction along which the boundaries are set to open.

        """

        boundary_couplings, boundary_indices = self.get_boundary_couplings(direction=direction)
        for _ in boundary_indices:
            self.set_coupling(0, _, by_symmetry=False)
            if isinstance(self, TightBindingModel):
                self.set_spin_orbit(0, np.ones(3), _, by_symmetry=False)
            else:
                self.set_DM([0, 0, 0], _, by_symmetry=False)

    def show_moments(self):
        """
        Prints the magnetic moments

        """

        struc = self.STRUC if self.supercell is None else self.supercell

        # TODO: make this for the supercell
        for _, site in enumerate(struc):
            print(f'Magnetic Moment on Site{_}:\t{site.properties["magmom"]}')

    def show_anisotropies(self):
        """
        Prints the single ion anisotropies.

        """

        for _, site in enumerate(self.STRUC):
            print(f'Single ion anisotropy on Site{_}:\t{site.properties["single_ion_anisotropy"]}')

    def write_cif(self, path=None):
        """
        Function that writes the pymatgen structure to a mcif file for visualization in e.g. VESTA

        Parameters
        ----------
        path : str
            Absolute path to the directory where the mcif file should be saved, e.g.
            '/home/user/material/material.mcif'
        """

        # get user's home directory
        home = str(Path.home())
        path = home + '/topwave_model.mcif' if path is None else path

        CifWriter(self.STRUC, write_magmoms=True).write_file(path)

    def get_set_couplings(self):
        """ Function that returns list of couplings that have been set and assigned a label.

        Returns
        -------
        set_couplings : list
            Couplings from self.CPLS that have been set and assigned a label via
            'set_coupling' or 'set_DM'.

        """

        set_couplings = []
        for cpl in self.CPLS:
            if cpl.label is not None or cpl.label_DM is not None:
                set_couplings.append(cpl)

        return set_couplings


class SpinWaveModel(ModelMixin):
    """
    Class for a Spin Wave Model.

    Methods
    -------
    get_classical_energy():
        Returns the classical ground state energy of the model.
    get_classical_groundstate():
        Adjusts the orientation of magnetic moments until a minimum in classical
        energy is reached.
    set_DM(D, symid, by_symmetry):
        Assign anti-symmetric exchange to a selection of couplings based on
        their symmetry index.
    set_single_ion_anisotropy(K, site_index, space_group):
        Assign single-ion anisotropy to site or selection thereof based on
        their symmetry index.
    """

    def get_classical_energy(self, per_spin=True):
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
        for coupling in self.CPLS:
            energy += coupling.get_energy()

        struc = self.STRUC if self.supercell is None else self.supercell

        # Zeeman energy and anisotropies
        for site in struc:
            magmom = site.properties['magmom']
            K = site.properties['single_ion_anisotropy']
            energy += magmom @ np.diag(K) @ magmom
            energy -= ModelMixin.muB * ModelMixin.g * (self.MF @ magmom)

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

        struc = self.STRUC if self.supercell is None else self.supercell

        moments = np.array([site.properties['magmom'] for site in struc], dtype=float)
        magnitudes = norm(moments, axis=1)

        # get initial moments
        x_0 = np.random.rand(struc.num_sites, 3) if random_init else moments

        res = minimize(SpinWaveModel.__get_classical_energy_wrapper, x_0, args=(magnitudes, self))

        # normalize the final configuration
        res.x = (res.x.reshape((-1, 3)).T / norm(res.x.reshape((-1, 3)), axis=1)).flatten(order='F')
        return res

    def set_DM(self, D, index, by_symmetry=True):
        """
        Assigns asymmetric exchange terms to a selection of couplings based
        on their symmetry. The vector that is passed is rotated according to
        the symmetries as well.

        Parameters
        ----------
        D : list
            Three-dimensional vector that gives the anti-symmetric part of
            the exchange.
        index : int
            Integer that corresponds to the symmetry index of a selection of
            couplings, or to the index if by_symmetry = False.
        by_symmetry : bool
            If true, index corresponds to the symmetry index of a selection of couplings.
            If false, it corresponds to the index.
        """
        # IDEA for the problem with implementation of future symbolic representation and labels
        # there's a problem when DM is initialized first on a bond. Maybe just check when DM is
        # set whether there's a J bond, and if not create one with 0 strength.
        if by_symmetry:
            indices = self.CPLS_as_df.index[self.CPLS_as_df['symid'] == index].tolist()
        else:
            indices = self.CPLS_as_df.index[self.CPLS_as_df.index == index].tolist()

        D = np.array(D, dtype=float)
        _ = indices[0]
        self.CPLS[_].DM = D
        self.CPLS_as_df.loc[_, 'DM'][0] = D[0]
        self.CPLS_as_df.loc[_, 'DM'][1] = D[1]
        self.CPLS_as_df.loc[_, 'DM'][2] = D[2]

        if by_symmetry:
            for _ in indices[1:]:
                Drot = self.CPLS[_].SYMOP.apply_rotation_only(D)
                self.CPLS[_].DM = Drot
                self.CPLS_as_df.loc[_, 'DM'][0] = Drot[0]
                self.CPLS_as_df.loc[_, 'DM'][1] = Drot[1]
                self.CPLS_as_df.loc[_, 'DM'][2] = Drot[2]

    def set_single_ion_anisotropy(self, K, site_index, space_group=None):
        """Assigns single-ion anisotropies to a selection of bonds based on their symmetry.

        Parameters
        ----------
        K : list
            Three-dimensional list for the three possible components of the single-ion anisotropy term.
        site_index : int
            Integer that corresponds to the index of a site. If space_group is not None, the term is added
            to all sites in the orbit of the provided site and transformed accordingly.
        space_group : int
            International number corresponding to a space group by which the orbit of the site is generated.
            Default is None.
        """

        if self.supercell is not None:
            logging.warning('Single-ion-anisotropies must be generated before the supercell'
                            'if you want them to be copied over.')

        K = np.array(K, dtype=float).reshape(3,)

        if space_group is not None:
            space_group = SpaceGroup.from_int_number(space_group)
            coords, ops = space_group.get_orbit_and_generators(self.STRUC[site_index].frac_coords)
            sites = []
            for coord in coords:
                cartesian_coord = self.STRUC.lattice.get_cartesian_coords(coord)
                sites.append(self.STRUC.get_sites_in_sphere(cartesian_coord, 1e-06)[0])
        else:
            sites = [self.STRUC[site_index]]
            ops = [SymmOp.from_rotation_and_translation(np.eye(3), np.zeros(3))]

        for site, op in zip(sites, ops):
            site.properties['single_ion_anisotropy'] = op.apply_rotation_only(K)




class TightBindingModel(ModelMixin):
    """
    Class for a tight-binding model.

    Parameters
    ----------
    struc : pymatgen.core.Structure
        pymatgen Structure that contains all the sites of the model.

    Attributes
    ----------
    spinful : bool
        Flag that specifies whether the model is spinless (fully spin polarized) or spinful.
        Setting an external magnetic field will automatically set it True. Default is False.

    Methods
    -------
    get_symbolic_hamiltonian():
        Returns a symbolic representation of the Hamiltonian.
    make_spinful():
        Adds spin degree of freedom to the Hamiltonian.
    set_onsite_term(strength, site_index, label, spin_matrix):
        Adds onsite energy terms to the given sites.
    set_spin_orbit(strength, matrix, index, by_symmetry, label):
        Adds a (generally) complex hopping term that couples spin-up
        and -down degrees of freedom.
    """

    def __init__(self, struc):
        super().__init__(struc)
        self.spinful = False

    def get_symbolic_hamiltonian(self):
        """ Uses sympy to construct and return a symbolic representation of
        the Hamiltonian.

        Returns
        -------
        symbolic_hamiltonian : sympy.matrices.dense.Matrix
            Symbolic Hamiltonian
        """

        symbolic_hamiltonian = sp.Matrix(np.zeros((self.N, self.N)))
        kx, ky, kz = sp.symbols('k_x k_y k_z', real=True)
        labels = []
        symbols = []
        for cpl in self.get_set_couplings():
            if cpl.label in labels:
                index = labels.index(cpl.label)
                symbol = symbols[index]
            else:
                labels.append(cpl.label)
                if np.imag(cpl.strength) == 0:
                    symbol = sp.Symbol(cpl.label, real=True)
                else:
                    symbol = sp.Symbol(cpl.label)
                symbols.append(symbol)
            fourier_coefficient = sp.exp(-sp.I * (cpl.R[0] * kx + cpl.R[1] * ky + cpl.R[2] * kz))
            symbolic_hamiltonian[cpl.I, cpl.J] += symbol * fourier_coefficient
            symbolic_hamiltonian[cpl.J, cpl.I] += (symbol * fourier_coefficient).conjugate()

        labels = [site.properties['onsite_label'] for site in self.STRUC]
        unique_labels = [None]
        for _, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                symbol = sp.Symbol(label, real=True)
                symbols.append(symbol)
                indices = [index for index in range(self.N) if labels[index] == label]
                for index in indices:
                    symbolic_hamiltonian[index, index] += symbol

        symbols = [kx, ky, kz] + symbols
        return sp.nsimplify(symbolic_hamiltonian), symbols

    def make_spinful(self):
        """Sets the spinful flag to True adding spin degree of freedom to the Hilbert space.
        """

        self.spinful = True

    def set_onsite_term(self, strength, site_index=None, label=None, spin_matrix=None):
        """Adds onsite term to the specified diagonal matrix element of the Hamiltonian.

        Parameters
        ----------
        strength : float
            Magnitude of the onsite term.
        site_index : int
            Site index (ordered as in self.STRUC) that the term will be added to.
            If None, the term will be added to all sites. Default is None.
        label : str
            A label that is used for the symbolic representation of the Hamiltonian. If None,
            an automatic label is generated. Default is None.
        spin_matrix : numpy.ndarray
            Two-by-two array in spin space for spin-dependant onsite terms, e.g. a spin-dependent
            staggered flux or a AFM Zeeman field. If not None, the model is made 'spinful'. Default is None.
        """

        if self.supercell is not None:
            logging.warning('Onsite terms must be generated before the supercell'
                            'if you want them to be copied over.')


        if site_index is None:
            site_indices = np.arange(self.N)
            auto_label = 'E_0'
        else:
            site_indices = [site_index]
            auto_label = f'E_{site_index}'

        for _ in site_indices:
            self.STRUC[_].properties['onsite_strength'] = strength
            self.STRUC[_].properties['onsite_label'] = label if label is not None else auto_label
            self.STRUC[_].properties['onsite_spin_matrix'] = spin_matrix

        if spin_matrix is not None:
            self.make_spinful()

    def show_onsite_terms(self):
        """
        Prints the single ion anisotropies.

        """

        for _, site in enumerate(self.STRUC):
            energy = site.properties['onsite_strength']
            spin = site.properties['onsite_spin_matrix']
            print(f'Onsite energy on Site{_}:\t{energy}\nSpin:\n{spin}')

    def set_spin_orbit(self, strength, vector, index, by_symmetry=True, label=None):
        """
        Sets a spin-orbit (hopping) term that couples the spin degrees of freedom.
        Automatically calls the 'make_spinful'-method.

        Parameters
        ----------
        strength : complex
            Strength of the spin-orbit interaction.
        vector : numpy.ndarray
            A vector the components of which corresponds to the coeffecients of a linear combination
            of Pauli matrices. The vector will be normalized.
        index : int
            Integer that corresponds to the symmetry index of a selection of
            couplings, or to the index if by_symmetry = False.
        by_symmetry : bool
            If true, index corresponds to the symmetry index of a selection of couplings.
            If false, it corresponds to the index.
        label : str
            Label for the exchange/hopping parameter that is used for the symbolic
            representation of the Hamiltonian. If None, a label is generated based
            on the index.

        """

        self.make_spinful()

        if by_symmetry:
            indices = self.CPLS_as_df.index[self.CPLS_as_df['symid'] == index].tolist()
        else:
            indices = self.CPLS_as_df.index[self.CPLS_as_df.index == index].tolist()

        vector = np.array(vector, dtype=float)
        vector = np.zeros(3, dtype=float) if norm(vector) == 0 else strength * vector / norm(vector)
        _ = indices[0]
        self.CPLS[_].DM = vector
        self.CPLS_as_df.loc[_, 'DM'][0] = vector[0]
        self.CPLS_as_df.loc[_, 'DM'][1] = vector[1]
        self.CPLS_as_df.loc[_, 'DM'][2] = vector[2]

        if by_symmetry:
            for _ in indices[1:]:
                Drot = self.CPLS[_].SYMOP.apply_rotation_only(vector)
                self.CPLS[_].DM = Drot
                self.CPLS_as_df.loc[_, 'DM'][0] = Drot[0]
                self.CPLS_as_df.loc[_, 'DM'][1] = Drot[1]
                self.CPLS_as_df.loc[_, 'DM'][2] = Drot[2]

