from itertools import product

import numpy as np
from numpy.linalg import eig, eigh, eigvals, inv, multi_dot, norm
from scipy.linalg import block_diag

from topwave import solvers
from topwave.model import Model, TightBindingModel

class Spec():
    """Class that contains the Hamiltonian, its spectrum and other quantities derived thereof.

    Parameters
    ----------
    model : model.Model
        The model of which the Hamiltonian is built.
    ks : numpy.ndarray
        Array of three-dimensional vectors in k-space at which the
        Hamiltonian is constructed.
    parallel : bool
        Whether to use multiprocessing or not. Default is False.

    Attributes
    ----------
    KS : numpy.ndarray
        This is where ks is stored.
    KS_xyz : numpy.ndarray
        Same as KS, but in cartesian coordinates in units of 1/Angstrom.
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
    parallel : bool
        This is where parallel is stored.
    S_perp : numpy.ndarray
        The neutron scattering cross section.

    Methods
    -------
    solve():
        Diagonalizes the Hamiltonian.

    """

    def __init__(self, model, ks, parallel=False):

        # store k-points
        self.KS = ks
        self.KS_xyz = 2 * np.pi * np.einsum('ka, ab -> kb', ks, inv(model.STRUC.lattice.matrix))

        # allocate memory for the Hamiltonian and its spectrum
        self.H = None
        self.DHDK = None
        self.OMEGA = None
        self.E = None
        self.psi = None
        self.SS = None
        self.N = len(model.STRUC)
        self.NK = len(ks)
        self.parallel = parallel
        self.S_perp = None

        # NOTE: think about the real implementation. Maybe two child classes of spec?
        if isinstance(model, TightBindingModel):
            self.H = self.get_tb_hamiltonian(model, self.KS)
            self.solve(eigh)
        # build Hamiltonian and diagonalize it
        else:
            self.H = self.get_sw_hamiltonian(model)
            self.solve(solvers.colpa)
            self.get_correlation_functions(model, parallel=self.parallel)

        # TODO: make switches for these so they aren't calculated all the time
        # compute the local spin-spin correlation functions
        #self.get_correlation_functions(model)

        # compute the tangent matrices of the hamiltonian and the Berry Curvature
        #self.DHDK = self.get_tangent_matrices(model)

        # compute the Berry curvature
        # self.get_berry_curvature()

    @staticmethod
    def get_tb_hamiltonian(model, ks):
        """ Function that builds the Hamiltonian for a tight-binding model.

        Parameters
        ----------
        model : topwave.model.Model
            The spin wave model that is used to construct the Hamiltonian.
        ks : numpy.ndarray
            Array of three-dimensional vectors in k-space at which the
            Hamiltonian is constructed.

        Returns
        -------
        The Hamiltonian of the model at the provided k-points.

        """

        N = len(model.STRUC)
        NK = len(ks)
        MAT = np.zeros((NK, N, N), dtype=complex)

        # construct matrix elements at each k-point
        for _, k in enumerate(ks):
            for cpl in model.get_set_couplings():
                # get the matrix elements from the couplings
                (A, inner) = cpl.get_tb_matrix_elements(k)

                MAT[_, cpl.I, cpl.J] += A
                MAT[_, cpl.J, cpl.I] += np.conj(A)

        # add onsite energy terms
        for _, site in enumerate(model.STRUC):
            MAT[:, _, _] += site.properties['onsite_energy']

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

    def get_spin_spin_expectation_val(self, model, ks, psi_k):
        """
        Calculates the local spin-spin expectation values for a given set of kpoints.

        Parameters
        ----------
        model : topwave.Model
            The model that is used to calculate the spectrum.
        ks : numpy.ndarray
            List of k-points. Shape is (NK, 3).
        psi_k : numpy.ndarray
            Wave functions used to calculate the spin-spin expectation values at the provided k-points.
            Shape is (NK, 2N, 2N).

        Returns
        -------
        SS_k : numpy.ndarray

        """

        # read out the number of k-points
        nk = len(ks)
        N = self.N

        # calculate the phase factors for all sites
        phases = np.zeros((nk, N), dtype=complex)
        us = np.zeros((N, 3), dtype=complex)
        for _, site in enumerate(model.STRUC.sites):
            mu = np.sqrt(norm(site.properties['magmom'] / 2))
            phases[:, _] = mu * np.exp(-1j * np.einsum('ki, i -> k', ks, site.frac_coords) * 2 * np.pi)
            us[_, :] = site.properties['Rot'][:, 0] + 1j * site.properties['Rot'][:, 1]

        phasesL = np.transpose(np.tile(np.concatenate((phases, phases), axis=1), (3, 3, 8, 1, 1)), (3, 2, 4, 0, 1))
        phasesR = np.conj(phasesL.swapaxes(1, 2))

        usL = np.transpose(np.tile(np.concatenate((us, np.conj(us)), axis=0), (2 * N, 3, 1, 1)), (0, 2, 3, 1))
        usL = np.tile(usL, (nk, 1, 1, 1, 1))
        usR = np.conj(np.transpose(usL, (0, 2, 1, 4, 3)))

        psiR = np.transpose(np.tile(psi_k, (3, 3, 1, 1, 1)), (2, 3, 4, 0, 1))
        psiL = np.conj(psiR.swapaxes(1, 2))

        S_k = np.sum(usL * phasesL * psiL, axis=2) * np.sum(usR * phasesR * psiR, axis=1)

        return S_k


        # # iterate over all possible combinations of spin-operator-pairs
        # for a, b in product(range(3), range(3)):
        #     # allocate memory for the two-operator matrices
        #     SS = np.zeros((2 * self.N, 2 * self.N), dtype=complex)
        #     # iterate over all the combinations of sites
        #     for i, j in product(range(self.N), range(self.N)):
        #         # construct the prefactor
        #         mu_i = norm(model.STRUC[i].properties['magmom'])
        #         mu_j = norm(model.STRUC[j].properties['magmom'])
        #         c = np.sqrt(mu_i * mu_j)
        #
        #         # construct the phase factor
        #         delta = model.STRUC[j].frac_coords - model.STRUC[i].frac_coords
        #         c_k = np.exp(-1j * (delta @ k) * 2 * np.pi)
        #
        #         # get the u-vectors
        #         u_i = model.STRUC[i].properties['Rot'][:, 0] + 1j * model.STRUC[i].properties['Rot'][:, 1]
        #         u_j = model.STRUC[j].properties['Rot'][:, 0] + 1j * model.STRUC[j].properties['Rot'][:, 1]
        #
        #         # calculate the two-operator matrix elements
        #         SS[i, j] = c * c_k * np.conj(u_i[a]) * np.conj(u_j[b])
        #         SS[i + self.N, j + self.N] = c * c_k * np.conj(u_i[a]) * u_j[b]
        #         SS[i, j + self.N] = c * c_k * u_i[a] * u_j[b]
        #         SS[i + self.N, j] = c * c_k * np.conj(u_i[a]) * np.conj(u_j[b])
        #
        #         # calculate the local spin-spin expectation values and save them
        #         SS_k[:, a, b] = np.diag(np.conj(psi_k.T) @ SS @ psi_k)
        #
        # return SS_k

    def get_correlation_functions(self, model, parallel=False):
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

        # don't use parallel (yet)
        if parallel:
            self.SS = np.zeros((self.NK, 2 * self.N, 3, 3), dtype=complex)
            for _, (k, psi_k) in enumerate(zip(self.KS, self.psi)):
                self.SS[_] = self.get_spin_spin_expectation_val(model, [k], [psi_k])
        else:
            self.SS = self.get_spin_spin_expectation_val(model, self.KS, self.psi)

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