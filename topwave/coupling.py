#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:32:35 2022

@author: niclas
"""

import numpy as np
from numpy.linalg import norm
import pandas as pd

from topwave.util import Pauli


class Coupling:
    """ Base class that holds information about a coupling between two sites
    
    Parameters
    ----------
    site1 : pymatgen.core.sites.PeriodicSite
        First site of the coupling.
    site2 : pymatgen.core.sites.PeriodicSite
        Second site of the coupling.
    cplid : int
        Index that enumerates all the couplings in the model. 
    symid : int
        Index that enumerates all symmetrically non-equivalent couplings
        in the model. 
    symop : pymatgen.core.operations.SymmOp
        Symmetry operation that connects self to the coupling with 
        symop = unity of the same symid group. 
    R : numpy.ndarray
        Three-dimensional vector that shows that describes the change of unit
        cell along the coupling. 
    
        
    Attributes
    ----------
    SITE1: pymatgen.core.sites.PeriodicSite
        This is where site1 is stored.
    SITE2: pymatgen.core.sites.PeriodicSite
        This is where site2 is stored.
    ID : int
        This is where cplid is stored.
    SYMID : int
        This is where symid is stored.
    SYMOP : pymatgen.core.operations.SymmOp
        This is where symop is stored.
    I : int
        Index that stores site1.properties['id'].
    J : int
        Index that stores site2.properties['id'].
    D : float
        Distance between the two sites of the coupling (in Angstrom)
    R : numpy.ndarray
        This is where R is stored.
    DELTA : numpy.ndarray
        Distance between two sites (in the same unit cell) in fractional
        coordinates. 
    strength : float
        Exchange/Hopping strength of the coupling
    DM : list
        Three dimensional vector specifying the anisotropic exchange of the coupling.
    spin_orbit : numpy.ndarray
        Complex two-by-two matrix specifying the mixing of the spin degrees of freedom.
    u : numpy.ndarray
        Three-dimensional vector related to the site1/site2 property 'Rot'
        that is used to construct the matrix elements in rotated local spin
        reference frames.
    v : numpy.ndarray
        Three-dimensional vector related to the site1/site2 property 'Rot'
        that is used to construct the matrix elements in rotated local spin
        reference frames.
    label : str
        Label that is used for the coupling part of the symbolic representation of the Hamiltonian.
    label_DM : str
        String that is used for the DM part of the symbolic representation of the Hamiltonian.
        
    Methods
    -------
    empty_df():
        Returns an empty dataframe with the right column labels for printing.
    get_df():
        Returns attributes of self as a pandas dataframe.
    get_energy():
        Returns the classical exchange energy of the coupling.
    get_exchange_matrix():
        Returns the exchange matrix of the coupling.
    get_label(label=None, by_symmetry=True):
        Generates labels for the symbolic representation of the Hamiltonian.
    get_uv():
        Constructs the u and v vectors when the system was magnetized.
    get_sw_matrix_elements(k):
        Generates matrix elements of the coupling at a given k-point.


    """

    def __init__(self, site1, site2, cplid, symid, symop, R):

        # store the input
        self.SITE1 = site1
        self.SITE2 = site2
        self.ID = cplid
        self.SYMID = symid
        self.SYMOP = symop
        self.I = site1.properties['id']
        self.J = site2.properties['id']
        self.D = site1.distance(site2, R)
        self.R = R
        self.DELTA = site2.frac_coords - site1.frac_coords
        self.strength = 0.
        self.DM = np.array([0., 0., 0.], dtype=float)
        self.spin_orbit = np.eye(2)
        self.label = None
        self.label_DM = None
        self.get_uv()
        df_data = [[self.SYMID, self.SYMOP.as_xyz_string(), self.DELTA, self.R, self.D, self.I,
                    str(self.SITE1.species), self.J, str(self.SITE2.species), self.strength, self.DM]]
        df_labels = ['symid', 'symop', 'delta', 'R', 'dist', 'i', 'at1', 'j', 'at2', 'strength', 'DM']
        self.DF = pd.DataFrame(df_data, columns=df_labels)

    def get_label(self, label=None, by_symmetry=True):
        """ Generates a label to the coupling to represent Hamiltonian symbolically.

        Parameters
        ----------
        label : str
            Label that is assigned to the coupling. If None, a label is assigned based on the
            (symmetry)-index of the coupling. Default is None.
        by_symmetry : bool
            If true the symmetry index will be used to assign a label. If false, the symmetry index
            and the index will be used. Default is True.
        """

        if label is None and by_symmetry is True:
            self.label = 'v_' + str(int(self.SYMID))
        elif label is None and by_symmetry is False:
            self.label = 'v_' + str(int(self.SYMID)) + str(int(self.ID))
        else:
            self.label = label

    def get_label_soc(self, label=None, by_symmetry=True):
        """ Generates a label to the coupling to represent Hamiltonian symbolically.

        Parameters
        ----------
        label : str
            Label that is assigned to the coupling for a given spin-orbit interaction.
            If None, a label is assigned based on the (symmetry)-index of the coupling. Default is None.
        by_symmetry : bool
            If true the symmetry index will be used to assign a label. If false, the symmetry index
            and the index will be used. Default is True.
        """

        if label is None and by_symmetry is True:
            self.label = 'l_' + str(int(self.SYMID))
        elif label is None and by_symmetry is False:
            self.label = 'l_' + str(int(self.SYMID)) + str(int(self.ID))
        else:
            self.label = label

    def get_label_DM(self, label=None, by_symmetry=True):
        """ Generates labels for the DM interaction strengths.

        label : str
            Label that is assigned to the coupling. If None, a label is assigned based on the
            (symmetry)-index of the coupling. Default is None.
        by_symmetry : bool
            If true the symmetry index will be used to assign a label. If false, the symmetry index
            and the index will be used. Default is True.
        """

        if label is None and by_symmetry is True:
            self.label_DM = 'D_' + str(int(self.SYMID))
        elif label is None and by_symmetry is False:
            self.label_DM = 'D_' + str(int(self.SYMID)) + str(int(self.ID))
        else:
            self.label_DM = label

    def get_uv(self):
        """ Constructs the u and v vector when the system was magnetized.
        

        """

        self.u1, self.u2, self.v1, self.v2 = [None] * 4
        try:
            R1 = self.SITE1.properties['Rot']
            R2 = self.SITE2.properties['Rot']
            u1 = R1[:, 0] + 1j * R1[:, 1]
            u2 = R2[:, 0] + 1j * R2[:, 1]
            v1 = R1[:, 2]
            v2 = R2[:, 2]
            self.u1, self.u2, self.v1, self.v2 = u1, u2, v1, v2
        except KeyError:
            pass

    def get_energy(self):
        """Returns the classical energy of the coupling.

        Returns
        -------
        energy : float
            The classical energy of the coupling.

        """

        exchange_matrix = self.get_exchange_matrix()
        S_1 = self.SITE1.properties['magmom']
        S_2 = self.SITE2.properties['magmom']
        energy = S_1 @ exchange_matrix @ S_2

        return energy

    def get_exchange_matrix(self):
        """Returns the exchange matrix.

        Returns
        -------
        exchange_matrix : numpy.ndarray
            Three-by-three numpy.ndarray that gives the exchange interaction of the bond.

        """

        exchange_matrix = np.array([[self.strength, self.DM[2], -self.DM[1]],
                                    [-self.DM[2], self.strength, self.DM[0]],
                                    [self.DM[1], -self.DM[0], self.strength]], dtype=complex)
        return exchange_matrix

    def get_fourier_coefficients(self, k):
        """
        Given a k-point this returns the Fourier coefficients for this bond,
        as well as the coefficients differentiated w.r.t. to all components of k.

        Parameters
        ----------
        k : numpy.ndarray
            Three-dimensional array corresponding to some k-point.
        Returns
        -------
        c_k : complex
            Fourier coefficient of the coupling at given k-point.
        inner : numpy.ndarray
            Derivatives of c_k w.r.t. to all components of the given k-point.
        """

        # two different choices of FT
        # c_k = np.exp(-1j * ((self.DELTA + self.R) @ k) * 2 * np.pi)
        c_k = np.exp(-1j * (self.R @ k) * 2 * np.pi)

        # inner derivative w.r.t. to k
        # inner = -1j * (self.DELTA + self.R) * 2 * np.pi
        inner = -1j * self.R * 2 * np.pi

        return c_k, inner

    def get_sw_matrix_elements(self, k):
        """ Constructs the matrix elements for the spin wave Hamiltonian.
        
        Parameters
        ----------
        k : numpy.ndarray
            Three-dimensional array corresponding to some k-point.

        Returns
        -------
        Matrix elements.

        """

        # construct the pre- and phase factors
        mu1 = norm(self.SITE1.properties['magmom'])
        mu2 = norm(self.SITE2.properties['magmom'])
        c = np.sqrt(mu1 * mu2) / 2.

        c_k, inner = self.get_fourier_coefficients(k)

        # constructs the exchange matrix
        Jhat = self.get_exchange_matrix()

        # construct the matrix elements
        # NOTE: Check redundancy for the a (and possibly b)-type matrix elements.
        A = c * c_k * (self.u1 @ Jhat @ np.conj(self.u2))
        Abar = c * np.conj(c_k) * (self.u1 @ Jhat @ np.conj(self.u2))

        CI = mu1 * (self.v1 @ Jhat @ self.v2)
        CJ = mu2 * (self.v1 @ Jhat @ self.v2)

        B = c * c_k * (self.u1 @ Jhat @ self.u2)
        Bbar = c * np.conj(c_k) * np.conj(self.u2 @ Jhat @ self.u1)

        return A, Abar, CI, CJ, B, Bbar, inner

    def get_tb_matrix_elements(self, k):
        """ Constructs the matrix elements for the tight-binding Hamiltonian.

        Parameters
        ----------
        k : numpy.ndarray
            Three-dimensional array corresponding to some k-point.

        Returns
        -------
        Matrix elements.

        """

        c_k, inner = self.get_fourier_coefficients(k)

        # construct the matrix elements
        A = c_k * self.strength

        # construct the spin-orbit coupling hoppings
        spin_orbit_term = 1j * Pauli(self.DM, normalize=False)

        return A, inner

    def get_spin_orbit_matrix_elements(self, k):
        """Creates the matrix elements of the tight-binding Hamiltonian that come from spin-orbit interation.

        Parameters
        ----------
        k : numpy.ndarray
            Three-dimensional array corresponding to some k-point.

        Returns
        -------
        Matrix elements.

        """

        c_k, inner = self.get_fourier_coefficients(k)

        # construct the spin-orbit coupling hoppings
        spin_orbit_term = 1j * c_k * Pauli(self.DM, normalize=False)

        return spin_orbit_term, inner

    def __repr__(self):
        return repr(self.DF)






    