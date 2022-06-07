#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:07:56 2022

@author: niclas
"""
from itertools import product
from pathlib import Path

from topwave import solvers
from topwave.coupling import Coupling
from topwave.util import rotate_vector_to_ez

import numpy as np
from numpy.linalg import eigvals, multi_dot, eig, eigh
import pandas as pd
from pymatgen.io.cif import CifWriter
from scipy.linalg import norm, block_diag
import sympy as sp
from tabulate import tabulate



class Model(object):
    """
    Base class that contains the physical model

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
    get_symbolic_hamiltonian():
        Returns a symbolic representation of the Hamiltonian
    """

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

        return sp.nsimplify(symbolic_hamiltonian)


class Spec(object):
    """
    Class that contains the Hamiltonian, its spectrum and other quantities derived thereof

    Parameters
    ----------
    model : model.Model
        The model of which the Hamiltonian is built.
    ks : numpy.ndarray
        Array of three-dimensional vectors in k-space at which the
        Hamiltonian is constructed.

    Attributes
    ----------
    KS : numpy.ndarray
        This is where ks is stored.
    H : numpy.ndarray
        This is where the matrix representation of the Hamiltonian is stored.
    DHDK : numpy.ndarray
        This is where the tangent matrices of MAT w.r.t. to k are stored.
    OMEGA : numpy.ndarray
        This is where the Berry Curvature is stored.
    E : numpy.ndarray
        This is where the eigenvalues of the Hamiltonian are stored.
    psi : numpy.ndarray
        This is where the eigenvectors of the Hamiltonian are stored.
    SS : numpy.ndarray
        This is where the spin-spin-correlation functions are stored
    N : int
        Number of magnetic sites in the model
    NK : int
        Number of k-points in ks

    Methods
    -------
    solve():
        Diagonalizes the bosonic Hamiltonian.
    """

    def __init__(self, model, ks):

        # store k-points
        self.KS = ks

        # allocate memory for the Hamiltonian and its spectrum
        self.H = None
        self.DHDK = None
        self.OMEGA = None
        self.E = None
        self.psi = None
        self.SS = None
        self.N = len(model.STRUC)
        self.NK = len(ks)

        # NOTE: think about the real implementation. Maybe two child classes of spec?
        if isinstance(model, TightBindingModel):
            self.H = self.get_tb_hamiltonian(model)
            self.solve(eigh)
        # build Hamiltonian and diagonalize it
        else:
            self.H = self.get_sw_hamiltonian(model)
            self.solve(solvers.colpa)

        # TODO: make switches for these so they aren't calculated all the time
        # compute the local spin-spin correlation functions
        #self.get_correlation_functions(model)

        # compute the tangent matrices of the hamiltonian and the Berry Curvature
        #self.DHDK = self.get_tangent_matrices(model)

        # compute the Berry curvature
        # self.get_berry_curvature()

    def get_tb_hamiltonian(self, model):
        """ Function that builds the Hamiltonian for a tight-binding model.

        Parameters
        ----------
        model : topwave.model.Model
            The spin wave model that is used to construct the Hamiltonian.

        Returns
        -------
        The Hamiltonian of the model at the provided k-points.

        """

        MAT = np.zeros((self.NK, self.N, self.N), dtype=complex)

        # construct matrix elements at each k-point
        for _, k in enumerate(self.KS):
            for cpl in model.get_set_couplings():
                # get the matrix elements from the couplings
                (A, inner) = cpl.get_tb_matrix_elements(k)

                MAT[_, cpl.I, cpl.J] += A
                MAT[_, cpl.J, cpl.I] += np.conj(A)

        return MAT

    def get_sw_hamiltonian(self, model):
        """
        Function that builds the Hamiltonian for the model at a set of
        given k-points.

        Parameters
        ----------
        model : topwave.model.Model
            The spin wave model that is used to construct the Hamiltonian.

        Returns
        -------
        The Hamiltonian of the model at the provided k-points.

        """

        MAT = np.zeros((self.NK, 2 * self.N, 2 * self.N), dtype=complex)

        # construct matrix elements at each k-point
        for _, k in enumerate(self.KS):
            for cpl in model.get_set_couplings():
                # get the matrix elements from the couplings
                (A, Abar, CI, CJ, B12, B21, inner) = cpl.get_sw_matrix_elements(k)

                MAT[_, cpl.I, cpl.J] += A
                MAT[_, cpl.J, cpl.I] += np.conj(A)
                MAT[_, cpl.I + self.N, cpl.J + self.N] += np.conj(Abar)
                MAT[_, cpl.J + self.N, cpl.I + self.N] += Abar

                MAT[_, cpl.I, cpl.I] -= CI
                MAT[_, cpl.J, cpl.J] -= CJ
                MAT[_, cpl.I + self.N, cpl.I + self.N] -= np.conj(CI)
                MAT[_, cpl.J + self.N, cpl.J + self.N] -= np.conj(CJ)

                # spurious
                MAT[_, cpl.I, cpl.J + self.N] += B12
                MAT[_, cpl.J, cpl.I + self.N] += B21
                MAT[_, cpl.J + self.N, cpl.I] += np.conj(B12)
                MAT[_, cpl.I + self.N, cpl.J] += np.conj(B21)

        # add the external magnetic field
        for _ in range(self.N):
            v = model.STRUC[_].properties['Rot'][:, 2]
            H_Zeeman = Model.muB * Model.g * (model.MF @ v)
            MAT[:, _, _] += H_Zeeman
            MAT[:, _ + self.N, _ + self.N] += H_Zeeman

        return MAT

    def get_tangent_matrices(self, model):
        """
        Similar to 'get_hamiltonian', but builds the tangent matrices instead

        Returns
        -------
        The three tangent matrices of the model w.r.t. the three components of
        the crystal momentum k.

        """

        DHDK = np.zeros((self.NK, 3, 2 * self.N, 2 * self.N), dtype=complex)

        # construct matrix elements at each k-point
        for _, k in enumerate(self.KS):
            for cpl in model.CPLS:
                # get the matrix elements from the couplings
                (A, Abar, CI, CJ, B12, B21, inner) = cpl.get_sw_matrix_elements(k)

                DHDK[_, :, cpl.I, cpl.J] += A * inner
                DHDK[_, :, cpl.J, cpl.I] += np.conj(A) * np.conj(inner)
                DHDK[_, :, cpl.I + self.N, cpl.J + self.N] += np.conj(Abar) * inner
                DHDK[_, :, cpl.J + self.N, cpl.I + self.N] += Abar * np.conj(inner)

                # TODO: add the derivatives of the B matrices

        return DHDK

    def get_omega_k(self, E_k, psi_k, dHdk_k):
        """
        Calculates the Berry curvature at one k-point

        Parameters
        ----------
        E_k : numpy.ndarray
            Vector that contains the energies of all the bands at k.
        psi_k : numpy.ndarray
            The eigenvectors at k.
        dHdk_k : numpy.ndarray
            The three tangent matrices of H at k.

        Returns
        -------
        The three components of the Berry curvature at k.

        """

        # number of bands
        N = E_k.shape[-1]

        # bosonic commutation relation matrix
        bos = block_diag(np.eye(self.N), -np.eye(self.N))

        # compute the square of all energy differences
        i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        denominator = np.square(E_k[i] - E_k[j])
        # set the diagonal elements to inf (we don't consider i=j elements)
        denominator[np.diag_indices(N)] = np.inf

        # check for degeneracies
        degenerate = np.isclose(denominator, 0.).any()

        if degenerate:
            raise ValueError

        else:
            Mx = np.conj(psi_k.T) @ dHdk_k[0] @  psi_k
            My = np.conj(psi_k.T) @ dHdk_k[1] @  psi_k
            Mz = np.conj(psi_k.T) @ dHdk_k[2] @  psi_k

            Omega_x = -2 * np.sum(np.imag((My * Mz.T) * np.reciprocal(denominator)), axis=1)
            Omega_y = -2 * np.sum(np.imag((Mz * Mx.T) * np.reciprocal(denominator)), axis=1)
            Omega_z = -2 * np.sum(np.imag((Mx * My.T) * np.reciprocal(denominator)), axis=1)

            Omega = np.array([Omega_x, Omega_y, Omega_z], dtype=float)

            return Omega

    def get_berry_curvature(self):
        """
        Calculates the Berry curvature at all (non degenerate) k-points.

        """

        # allocate memory (a nan array)
        self.OMEGA = np.zeros((self.NK, 3, 2 * self.N), dtype=float)
        self.OMEGA[:] = np.nan
        for _, (k, E_k, psi_k, dHdk_k) in enumerate(zip(self.KS, self.E, self.psi, self.DHDK)):
            # check if the energies are degenerate
            try:
                self.OMEGA[_] = self.get_omega_k(E_k, psi_k, dHdk_k)
            except:
                s = f'Spectrum is degenerate at {k}{E_k}. Berry Curvature is not well-defined at that point.'
                print(s)

    def get_spin_spin_expectation_val(self, model, k, psi_k):
        """
        Calculates the local spin-spin expectation values at one k-point.

        Returns
        -------
        SS_k

        """

        # allocate memory for the output
        SS_k = np.zeros((2 * self.N, 3, 3), dtype=complex)
        # iterate over all possible combinations of spin-operator-pairs
        for a, b in product(range(3), range(3)):
            # allocate memory for the two-operator matrices
            SS = np.zeros((2 * self.N, 2 * self.N), dtype=complex)
            # iterate over all the combinations of sites
            for i, j in product(range(self.N), range(self.N)):
                # construct the prefactor
                mu_i = norm(model.STRUC[i].properties['magmom'])
                mu_j = norm(model.STRUC[j].properties['magmom'])
                c = np.sqrt(mu_i * mu_j)

                # construct the phase factor
                delta = model.STRUC[j].frac_coords - model.STRUC[i].frac_coords
                c_k = np.exp(-1j * (delta @ k) * 2 * np.pi)

                # get the u-vectors
                u_i = model.STRUC[i].properties['Rot'][:, 0] + 1j * model.STRUC[i].properties['Rot'][:, 1]
                u_j = model.STRUC[j].properties['Rot'][:, 0] + 1j * model.STRUC[j].properties['Rot'][:, 1]

                # calculate the two-operator matrix elements
                SS[i, j] = c * c_k * np.conj(u_i[a]) * np.conj(u_j[b])
                SS[i + self.N, j + self.N] = c * c_k * np.conj(u_i[a]) * u_j[b]
                SS[i, j + self.N] = c * c_k * u_i[a] * u_j[b]
                SS[i + self.N, j] = c * c_k * np.conj(u_i[a]) * np.conj(u_j[b])

                # calculate the local spin-spin expectation values and save them
                SS_k[:, a, b] = np.diag(np.conj(psi_k.T) @ SS @ psi_k)

        return SS_k

    def get_correlation_functions(self, model):
        """
        Calculates the spin-spin correlation function for all k-points

        Parameters
        ----------
        model : topwave.Model
            The model that is used to calculate the spectrum.

        Returns
        -------
        None.

        """

        self.SS = np.zeros((self.NK, 2 * self.N, 3, 3), dtype=complex)
        for _, (k, psi_k) in enumerate(zip(self.KS, self.psi)):
            self.SS[_] = self.get_spin_spin_expectation_val(model, k, psi_k)

    def solve(self, solver):
        """
        Diagonalizes the bosonic Hamiltonian.

        Parameters
        ----------
        solver : function
            A function that takes a Hamiltonian, and returns its eigenvalues and vectors.

        Returns
        -------
        Eigenvalues and Vectors.

        """

        # allocate memory for the output
        E = np.zeros(self.H.shape[0:2])  # complex for white alg.
        psi = np.zeros(self.H.shape, dtype=complex)

        # diagonalize the Hamiltonian at each k-point
        for _, k in enumerate(self.KS):
            try:
                E[_], psi[_] = solver(self.H[_])
            except:
                s = 'Hamiltonian is not positive-definite at k = (%.3f, %.3f' \
                    ', %.3f). Adding small epsilon and trying again.' % tuple(k)
                print(s)
                try:
                    epsilon = np.sort(np.real(eigvals(self.H[_]))) + 0.0000001
                    # epsilon = 0.1
                    H_shftd = self.H[_] + np.eye(self.H.shape[1]) * epsilon
                    E[_], psi[_] = solver(H_shftd)
                except:
                    s = 'Diagonalization failed! Check classical ground state' \
                        ' or try different method for approximate' \
                        'diagonalization.'
                    raise TypeError(s)

        # save the eigenvalues and vectors and return them
        self.E = E
        self.psi = psi

        return E, psi

    def wilson_loop(self, occ):
        """
        Calculates the Wilson loop along the provided k-points

        Parameters:
        ----------
            occ : list
            List of integers specifying the band indices of all occupied bands.

        Returns
        -------
        Eigenvalues of the Wilson loop operator

        """

        # TODO: maybe make a check, that k-points form a closed loop

        # bosonic commutation relation matrix
        bos = block_diag(np.eye(self.N), -np.eye(self.N))

        # construct F = <m_k+1|bos|n_k> for each k-point
        psi_left = np.roll(np.swapaxes(np.conj(self.psi), 1, 2), 1, axis=0)
        psi_right = bos @ self.psi

        # choose only the occupied bands
        psi_left = psi_left[:, occ, :]
        psi_right = psi_right[:, :, occ]
        F = np.einsum('knm, kml -> knl', psi_left, psi_right)

        # compute the product W = F_k+N F_k+N-1 ... F_k,
        W = multi_dot(F[:, ...])

        # compute the Wilson loop operators spectrum
        lamda, v = eig(W)

        # calculate the phase angle and sort the eigenvalues. numpy.angle() has
        # target space [-pi, pi). Shift to [0, 2pi]!
        args = np.sort(np.angle(-lamda) + np.pi)

        self.wannier_center = args

    def wilson_loop_test(self, occ):
        """
        Calculates the Wilson loop along the provided k-points

        Parameters:
        ----------
            occ : list
            List of integers specifying the band indices of all occupied bands.

        Returns
        -------
        Eigenvalues of the Wilson loop operator

        """

        # TODO: maybe make a check, that k-points form a closed loop

        # construct F = <m_k+1|bos|n_k> for each k-point
        psi_left = np.roll(np.swapaxes(np.conj(self.psi), 1, 2), 1, axis=0)
        psi_right = self.psi

        N = psi_right.shape[-1]//2

        # choose only the occupied bands
        psi_left = psi_left[:, occ, :N]
        psi_right = psi_right[:, :N, occ]
        F = np.einsum('knm, kml -> knl', psi_left, psi_right)

        # compute the product W = F_k+N F_k+N-1 ... F_k,
        W = multi_dot(F[:-1, ...])

        # compute the Wilson loop operators spectrum
        lamda, v = eig(W)

        # calculate the phase angle and sort the eigenvalues. numpy.angle() has
        # target space [-pi, pi). Shift to [0, 2pi]!
        args = np.sort(np.angle(-lamda) + np.pi)

        self.wannier_center = args

    def get_berry_curvature_test(self):
        """
        Calculates the Berry curvature at all (non degenerate) k-points.

        """

        # allocate memory (a nan array)
        self.OMEGA = np.zeros((self.NK, 3, self.N), dtype=float)
        self.OMEGA[:] = np.nan
        for _, (k, E_k, psi_k, dHdk_k) in enumerate(zip(self.KS, self.E[:, :self.N], self.psi[:, :self.N, :self.N], self.DHDK[:, :, :self.N, :self.N])):
            # check if the energies are degenerate
            try:
                self.OMEGA[_] = self.get_omega_k(E_k, psi_k, dHdk_k)
            except:
                s = f'Spectrum is degenerate at {k}{E_k}. Berry Curvature is not well-defined at that point.'
                print(s)