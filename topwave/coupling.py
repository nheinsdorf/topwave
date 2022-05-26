#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:32:35 2022

@author: niclas
"""

import numpy as np
from numpy.linalg import norm, matrix_power
import pandas as pd


class Coupling(object):
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
    JH : float
        Heisenberg exchange of the coupling
    u : numpy.ndarray
        Three-dimensional vector related to the site1/site2 property 'Rot'
        that is used to construct the matrix elements in rotated local spin
        reference frames.
    v : numpy.ndarray
        Three-dimensional vector related to the site1/site2 property 'Rot'
        that is used to construct the matrix elements in rotated local spin
        reference frames.
        
    Methods
    -------
    empty_df():
        Returns an empty dataframe with the right column labels for printing.
    get_df():
        Returns attributes of self as a pandas dataframe.
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
        self.JH = 0.
        self.DM = np.array([0., 0., 0.], dtype=float)
        self.u1, self.u2, self.v1, self.v2 = [None]*4
        self.DF = pd.DataFrame([[self.SYMID, self.SYMOP.as_xyz_string(), self.DELTA, self.R, self.D, self.I,
                                 str(self.SITE1.species), self.J, str(self.SITE2.species), self.JH, self.DM]],
                               columns=['symid', 'symop', 'delta', 'R', 'dist', 'i', 'at1', 'j', 'at2', 'Heis.', 'DM'])

    def get_uv(self):
        """ Constructs the u and v vector when the system was magnetized.
        

        """

        R1 = self.SITE1.properties['Rot']
        R2 = self.SITE2.properties['Rot']
        u1 = R1[:, 0] + 1j * R1[:, 1]
        u2 = R2[:, 0] + 1j * R2[:, 1]
        v1 = R1[:, 2]
        v2 = R2[:, 2]

        self.u1, self.u2, self.v1, self.v2 = u1, u2, v1, v2

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
        Jhat = np.array([[self.JH, self.DM[2], -self.DM[1]],
                         [-self.DM[2], self.JH, self.DM[0]],
                         [self.DM[1], -self.DM[0], self.JH]], dtype=complex)

        # construct the matrix elements
        A = c * c_k * (self.u1 @ Jhat @ np.conj(self.u2))
        Abar = c * np.conj(c_k) * (self.u1 @ Jhat @ np.conj(self.u2))

        CI = mu1 * (self.v1 @ Jhat @ self.v2)
        CJ = mu2 * (self.v1 @ Jhat @ self.v2)

        # spurious
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
        A = c_k * (self.u1 @ Jhat @ np.conj(self.u2))
        Abar = c * np.conj(c_k) * (self.u1 @ Jhat @ np.conj(self.u2))

        CI = mu1 * (self.v1 @ Jhat @ self.v2)
        CJ = mu2 * (self.v1 @ Jhat @ self.v2)

        # spurious
        B = c * c_k * (self.u1 @ Jhat @ self.u2)
        Bbar = c * np.conj(c_k) * np.conj(self.u2 @ Jhat @ self.u1)

        return A, Abar, CI, CJ, B, Bbar, inner

    @staticmethod
    def rotate_vector_to_ez(v):
        """ Creates a 3x3 rotation matrix R with R v = [0, 0, 1]


                Parameters
                ----------
                v : numpy.ndarray
                    Three-dimensional vector.

                Returns
                -------
                numpy.ndarray
                    3x3 rotation matrix R with R v = [0, 0, 1].

                """

        v = np.array(v, dtype=float) / norm(v)
        e3 = v
        if np.isclose(np.abs(v), [1, 0, 0], atol=0.00001).all():
            e2 = np.array([0, 0, 1])
        else:
            e2 = np.cross(v, [1, 0, 0])
        e2 = e2 / norm(e2)
        e1 = np.cross(e2, e3)

        return np.array([e1, e2, e3]).T

    @staticmethod
    def align_unit_vectors(v1, v2=None):
        """ Creates a 3x3 rotation matrix R with R v1 = v2
        # SPURIOUS!!!!

        Parameters
        ----------
        v1 : numpy.ndarray
            Three-dimensional vector.
        v2 : numpy.ndarray
            Three-dimensional vector. If None v2 = [0, 0, 1]. Default is None.

        Returns
        -------
        numpy.ndarray
            3x3 rotation matrix R with Rv1 = v2.

        """

        v1 = np.array(v1, dtype=float)
        if v2 is None:
            v2 = np.array([0, 0, 1], dtype=float)
        else:
            v2 = np.array(v2)

        # make them unit vectors
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)

        # and calculate their dot product
        dot = v1 @ v2

        # check whether v1 and v2 are (anti-)parallel
        if np.isclose(dot, 1., atol=0.00001):
            return np.eye(3, dtype=float)

        elif np.isclose(dot, -1., atol=0.00001):
            # get the first non-zero element of v2
            idx = np.argmin(v2 == 0)
            # construct v_orth which is orthogonal to ez using the scalar product
            v_orth = np.ones(3)
            v_orth[idx] = -v2[np.delete(np.arange(3), idx)].sum()
            v_orth = v_orth / norm(v_orth)
            # calculate rotational matrix
            return -np.eye(3) + 2 * np.outer(v_orth, v_orth)

        else:
            angle = np.arccos(v1 @ v2) / 2 / np.pi * 360
            print(angle)
            cross = np.cross(v1, v2)
            skew_mat = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
            return np.eye(3) + skew_mat + matrix_power(skew_mat, 2) * (1 / (1 + dot))

    def __repr__(self):
        return repr(self.DF)






    