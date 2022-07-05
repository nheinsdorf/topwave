#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:07:56 2022

@author: niclas
"""
from pathlib import Path

import numpy as np
from numpy.linalg import eig, eigh, eigvals, multi_dot
import pandas as pd
from pymatgen.io.cif import CifWriter
from scipy.linalg import block_diag, norm
import sympy as sp
from tabulate import tabulate

from topwave.coupling import Coupling
from topwave.util import rotate_vector_to_ez

class Model():
    """Base class that contains the physical model.

    Parameters
    ----------
    struc : pymatgen.core.Structure
        pymatgen Structure that contains all the magnetic sites of the model

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


    Methods
    -------
    generate_couplings(maxdist, sg=None):
        Given a maximal distance (in Angstrom) all periodic bonds are
        generated and grouped by symmetry based on the provided sg.
    show_couplings():
        Prints the couplings.
    set_coupling(J, symid):
        Assign Heisenberg Exchange or hopping amplitude terms to a collection of couplings based
        on their symmetry index.
    set_field(B):
        Apply an external magnetic field that is stored in self.MF
    set_moments(gs):
        Provide a classical magnetic ground state.
    show_moments():
        Prints the magnetic moments.
    write_cif():
        A function that uses pymatgen functionality to save Model.STRUC
        as a mcif file to be visualized in e.g. VESTA

    """

    # maybe i should save these in topwave itself
    muB = 0.057883818066000  # given in meV/Tesla
    g = 2  # for now let's fix the g-tensor to 2 (as a number)

    def __init__(self, struc):

        # save the input structure
        self.STRUC = struc
        # and label the sites within struc by an index
        for _, site in enumerate(self.STRUC):
            site.properties['id'] = _
            site.properties['onsite_energy_label'] = None
            site.properties['onsite_energy'] = 0

        # count the number of magnetic sites and save
        self.N = len(self.STRUC)

        # allocate an empty list for the couplings and a dataframe (for easy access and printing)
        self.CPLS = None
        self.CPLS_as_df = None
        self.reset_all_couplings()

        # put zero magnetic field
        self.MF = np.zeros(3, dtype=float)

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

    def show_couplings(self):
        """
        Prints the couplings

        """

        print(tabulate(self.CPLS_as_df, headers='keys', tablefmt='github',
                       showindex=True))

    def set_coupling(self, strength, index, by_symmetry=True, label=None):
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
        label : str
            Label for the exchange/hopping parameter that is used for the symbolic
            representation of the Hamiltonian. If None, a label is generated based
            on the index.

        """

        if by_symmetry:
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
        be translated to a Zeeman term mu_B B dot S

        Parameters
        ----------
        direction : list
            Three-dimensional vector that gives direction of an external magnetic field.
        magnitude : float
            Strength of the external magnetic field.

        """

        field = np.real(magnitude) * np.array(direction, dtype=float) / norm(direction)
        self.MF[0], self.MF[1], self.MF[2] = tuple(field.tolist())

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

        directions = np.array(directions, dtype=float).reshape((self.N, 3))
        magnitudes = np.array(magnitudes, dtype=float).reshape((self.N,))
        for _, (direction, magnitude) in enumerate(zip(directions, magnitudes)):
            # rotate into cartesian coordinates and normalize it
            moment = self.STRUC.lattice.matrix.T @ direction
            moment = moment / norm(moment)

            # calculate the rotation matrix that rotates the spin to the quantization axis
            self.STRUC[_].properties['Rot'] = rotate_vector_to_ez(moment)
            # stretch it to match the right magnetic moment and save it
            self.STRUC[_].properties['magmom'] = moment * magnitude

        # extract the u- and v-vectors from the rotation matrix
        for cpl in self.CPLS:
            cpl.get_uv()

    def show_moments(self):
        """
        Prints the magnetic moments

        """

        for _, site in enumerate(self.STRUC):
            print(f'Magnetic Moment on Site{_}:\t{site.properties["magmom"]}')

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


class SpinWaveModel(Model):
    """
    Class for a Spin Wave Model.

    Methods
    -------
    set_DM(D, symid):
        Assign anti-symmetric exchange to a selection of couplings based on
        their symmetry index.
    """

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
        # set whether theres a J bond, and if not create one with 0 strength.
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


class TightBindingModel(Model):
    """
    Class for a tight-binding model.

    Methods
    -------
    set_onsite_energy(onsite_energy, site_index, label):
        Adds onsite energy terms to the given sites.
    get_symbolic_hamiltonian():
        Returns a symbolic representation of the Hamiltonian
    """

    def set_onsite_energy(self, onsite_energy, site_index=None, label=None):
        """ Adds onsite term to the specified diagonal matrix element of the Hamiltonian.

        Parameters
        ----------
        onsite_energy : float
            Magnitude of the onsite term.
        site_index : int
            Site index (ordered as in self.STRUC) that the term will be added to.
            If None, the term will be added to all sites. Default is None.
        label : str
            A label that is used for the symbolic representation of the Hamiltonian. If None,
            an automatic label is generated. Default is None.
        """

        if site_index is None:
            site_indices = np.arange(self.N)
            auto_label = 'E_0'
        else:
            site_indices = [site_index]
            auto_label = f'E_{site_index}'

        for _ in site_indices:
            self.STRUC[_].properties['onsite_energy'] = onsite_energy
            self.STRUC[_].properties['onsite_energy_label'] = label if label is not None else auto_label

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

        labels = [site.properties['onsite_energy_label'] for site in self.STRUC]
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


