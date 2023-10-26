from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy.linalg import eigh, eigvals, inv, multi_dot

from topwave import solvers
from topwave.constants import G_LANDE, MU_BOHR
from topwave.fourier_coefficients import get_periodic_fourier_coefficient, get_periodic_fourier_derivative
from topwave.set_of_kpoints import SetOfKPoints
from topwave.model import Model, SpinWaveModel, TightBindingModel
from topwave.topology import get_berry_phase, get_fermionic_wilson_loop
from topwave.types import IntVector, RealList, SquareMatrix, VectorList
from topwave.util import format_input_vector, format_kpoints, pauli

__all__ = ["Spec"]


# TODO: Make the constructor method an instance attribute and the get_..._ham static methods
# also get rid of 'diagonalize', but make the user use a static method so that spec class
# can always be expected to have energies and eigenvalues. MAYBE???
# Make a copy method for the model so you only store a copy of it.
@dataclass(slots=True)
class Spec:
    """Computes the spectrum of a model for a given set of k-points."""

    model: Model
    kpoints: VectorList | SetOfKPoints
    energies: npt.NDArray[np.float64] = field(init=False)
    hamiltonian: npt.NDArray[np.float64] = field(init=False)
    kpoints_xyz: npt.NDArray[np.float64] = field(init=False)
    psi: npt.NDArray[np.float64] = field(init=False)

    # TODO: refactor kpoints to kpoints
    def __post_init__(self) -> None:
        # compute this only when you need it e.g. neutron
        # self.kpoints_xyz = 2 * np.pi * np.einsum('ka, ab -> kb', self.kpoints, inv(self.model.structure.lattice.matrix))

        if not isinstance(self.kpoints, SetOfKPoints):
            self.kpoints = SetOfKPoints(self.kpoints)

        match self.model.get_type():
            case 'spinwave':
                constructor, solver = self.get_spinwave_hamiltonian, solvers.colpa
            case 'tightbinding':
                constructor, solver = self.get_tightbinding_hamiltonian, eigh

        self.hamiltonian = constructor(self.model, self.kpoints)

        self.energies, self.psi = self.solve(solver)


    def get_density_of_states(self,
                              energies: RealList,
                              broadening: float = 0.02) -> RealList:
        """Computes the density of states:

        Parameters
        ----------
        energies: RealList
            Energies at which the density of states is calculated.
        broadening: float
            Broadening of the spectral density. Default is 0.02.

        Returns
        -------
        RealList
            The density of states.

        See Also
        --------
        :class:`topwave.topology.get_berry_phase`


        Examples
        --------
        Calculate some DOS.

        """

        density_of_states = []
        for energy in energies:
            density_of_states.append(np.sum(self.get_spectral_density(energy, broadening)))
        return np.array(density_of_states, dtype=np.float64)

    @staticmethod
    def get_spinwave_hamiltonian(
            model: SpinWaveModel,
            kpoints: VectorList | SetOfKPoints) -> npt.NDArray[np.complex128]:
        """Constructs the spin wave Hamiltonian for a set of given k-points."""


        kpoints = format_kpoints(kpoints)

        k_dependence = get_periodic_fourier_coefficient

        dim = len(model.structure)
        matrix = np.zeros((len(kpoints), 2 * dim, 2 * dim), dtype=complex)

        # construct matrix elements at each k-point
        for coupling in model.get_set_couplings():

            i, j = coupling.site1.properties['index'], coupling.site2.properties['index']
            fourier_coefficients = k_dependence(coupling, kpoints)
            # get the matrix elements from the couplings
            (A, Abar, CI, CJ, B, Bbar) = coupling.get_spinwave_matrix_elements()

            matrix[:, i, j] += fourier_coefficients * A
            matrix[:, j, i] += np.conj(fourier_coefficients * A)
            matrix[:, i + dim, j + dim] += np.conj(np.conj(fourier_coefficients) * Abar)
            matrix[:, j + dim, i + dim] += np.conj(fourier_coefficients) * Abar

            matrix[:, i, i] -= CI
            matrix[:, j, j] -= CJ
            matrix[:, i + dim, i + dim] -= np.conj(CI)
            matrix[:, j + dim, j + dim] -= np.conj(CJ)

            matrix[:, i, j + dim] += fourier_coefficients * B
            matrix[:, j, i + dim] += np.conj(fourier_coefficients) * Bbar
            matrix[:, j + dim, i] += np.conj(fourier_coefficients * B)
            matrix[:, i + dim, j] += np.conj(np.conj(fourier_coefficients) * Bbar)

        # add single ion anisotropies
        for _ in range(dim):
            u = model.structure[_].properties['Rot'][:, 0] + 1j * model.structure[_].properties['Rot'][:, 1]
            v = model.structure[_]
            # K = np.diag(model.structure[_].properties['onsite_vector'])
            # this constructs an interaction matrix with a principal axis along the onsite vector
            K = np.linalg.norm(model.structure[_].properties['onsite_vector'])
            easy_axis = format_input_vector(model.structure[_].properties['onsite_vector'], 1)
            onsite_exchange_matrix = -K * np.outer(easy_axis, easy_axis) \
                                     + model.structure[_].properties['onsite_matrix']

            matrix[:, _, _] += u @ onsite_exchange_matrix @ np.conj(u)
            matrix[:, _ + dim, _ + dim] += np.conj(u @ onsite_exchange_matrix @ np.conj(u))
            matrix[:, _, _ + dim] += u @ onsite_exchange_matrix @ u
            matrix[:, _ + dim, _] += np.conj(u @ onsite_exchange_matrix @ u)

        # add the external magnetic field
        for _ in range(dim):
            v = model.structure[_].properties['Rot'][:, 2]
            H_Zeeman = MU_BOHR * G_LANDE * np.dot(model.zeeman, v)
            matrix[:, _, _] += H_Zeeman
            matrix[:, _ + dim, _ + dim] += H_Zeeman

        return matrix

    @staticmethod
    def get_tightbinding_hamiltonian(
            model: TightBindingModel,
            kpoints: VectorList | SetOfKPoints) -> SquareMatrix:
        """Constructs the Tight Binding Hamiltonian for a given set of k-points.


        Parameters
        ----------
        model: TightBindingModel
            The model of that is used to calculate the spectrum.
        kpoints: VectorList | SetOfKPoints
            The kpoints at which the Hamiltonian or its derivatives are constructed.

        Returns
        -------
        SquareMatrix
            The Hamiltonian.

        """

        kpoints = format_kpoints(kpoints)

        k_dependence = get_periodic_fourier_coefficient

        nums_orbitals = [site.properties['orbitals'] for site in model.structure]
        dimension = sum(nums_orbitals)
        combined_index_nodes = np.concatenate(([0], np.cumsum(nums_orbitals)[:-1]), dtype=np.int64)
        matrix = np.zeros((len(kpoints), dimension, dimension), dtype=complex)

        # construct matrix elements at each k-point
        # for _, kpoint in enumerate(kpoints):
        for coupling in model.get_set_couplings():
            i = combined_index_nodes[coupling.site1.properties['index']] + coupling.orbital1
            j = combined_index_nodes[coupling.site2.properties['index']] + coupling.orbital2

            # get the matrix elements from the couplings
            # A, inner = coupling.get_tightbinding_matrix_elements(kpoint)
            A = k_dependence(coupling, kpoints) * coupling.get_tightbinding_matrix_elements()
            matrix[:, i, j] += A
            matrix[:, j, i] += np.conj(A)

        for site_index, site in enumerate(model.structure):
            for orbital_index, onsite_scalar in enumerate(np.array(site.properties['onsite_scalar']).reshape((-1,))):
                i = combined_index_nodes[site_index] + orbital_index
                matrix[:, i, i] += onsite_scalar

        # add spin degrees of freedom
        if model.check_if_spinful():
            matrix = np.kron(matrix, np.eye(2))

            for site_index, site in enumerate(model.structure):
                # for orbital_index, onsite_vector in enumerate(np.array(site.properties['onsite_vector']).reshape((-1, 3))):
                for orbital_index in range(site.properties['orbitals']):
                    i = combined_index_nodes[site_index] + orbital_index
                    # add zeeman term
                    matrix[:, 2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += MU_BOHR * G_LANDE * pauli(model.zeeman, normalize=False)

                    # add onsite term (for all orbitals the same at the moment)
                    matrix[:, 2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += pauli(site.properties['onsite_vector'], normalize=False)

            # add spin-orbit term
            # for _, kpoint in enumerate(kpoints):
            for coupling in model.get_set_couplings():
                i = combined_index_nodes[coupling.site1.properties['index']] + coupling.orbital1
                j = combined_index_nodes[coupling.site2.properties['index']] + coupling.orbital2
                spin_orbit_term = np.einsum('c, nm -> cnm', k_dependence(coupling, kpoints), coupling.get_spin_orbit_matrix_elements())
                matrix[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += spin_orbit_term
                matrix[:, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conj(spin_orbit_term.swapaxes(1, 2))

            if model._is_spin_polarized:
                return matrix[:, ::2, ::2]
        return matrix

    @staticmethod
    def get_tightbinding_derivative(
            model: TightBindingModel,
            kpoints: VectorList | SetOfKPoints,
            derivative: str) -> SquareMatrix:
        """Constructs the derivative of a Tight Binding Hamiltonian for a given set of k-points.

        TODO: update for multi-orbital
        Parameters
        ----------
        model: TightBindingModel
            The model of that is used to calculate the spectrum.
        kpoints: VectorList | SetOfKPoints
            The kpoints at which the Hamiltonian or its derivatives are constructed.
        derivative: str
            String that indicates which derivative should be constructed. Options are 'x', 'y' or 'z'.

        Returns
        -------
        SquareMatrix
            The derivative of the Hamiltonian.

        """

        kpoints = format_kpoints(kpoints)

        k_dependence = partial(get_periodic_fourier_derivative, direction=derivative)

        nums_orbitals = [site.properties['orbitals'] for site in model.structure]
        dimension = sum(nums_orbitals)
        combined_index_nodes = np.concatenate(([0], np.cumsum(nums_orbitals)[:-1]), dtype=np.int64)
        matrix = np.zeros((len(kpoints), dimension, dimension), dtype=complex)

        # construct matrix elements at each k-point
        # for _, kpoint in enumerate(kpoints):
        for coupling in model.get_set_couplings():
            i = combined_index_nodes[coupling.site1.properties['index']] + coupling.orbital1
            j = combined_index_nodes[coupling.site2.properties['index']] + coupling.orbital2

            # get the matrix elements from the couplings
            # A, inner = coupling.get_tightbinding_matrix_elements(kpoint)
            A = k_dependence(coupling, kpoints) * coupling.get_tightbinding_matrix_elements()
            matrix[:, i, j] += A
            matrix[:, j, i] += np.conj(A)

        for site_index, site in enumerate(model.structure):
            for orbital_index, onsite_scalar in enumerate(np.array(site.properties['onsite_scalar']).reshape((-1,))):
                i = combined_index_nodes[site_index] + orbital_index
                matrix[:, i, i] += onsite_scalar

        # add spin degrees of freedom
        if model.check_if_spinful():
            matrix = np.kron(matrix, np.eye(2))

            for site_index, site in enumerate(model.structure):
                # for orbital_index, onsite_vector in enumerate(np.array(site.properties['onsite_vector']).reshape((-1, 3))):
                for orbital_index in range(site.properties['orbitals']):
                    i = combined_index_nodes[site_index] + orbital_index
                    # add zeeman term
                    matrix[:, 2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += MU_BOHR * G_LANDE * pauli(model.zeeman,
                                                                                               normalize=False)

                    # add onsite term (for all orbitals the same at the moment)
                    matrix[:, 2 * i: 2 * i + 2, 2 * i: 2 * i + 2] += pauli(site.properties['onsite_vector'],
                                                                           normalize=False)

            # add spin-orbit term
            # for _, kpoint in enumerate(kpoints):
            for coupling in model.get_set_couplings():
                i = combined_index_nodes[coupling.site1.properties['index']] + coupling.orbital1
                j = combined_index_nodes[coupling.site2.properties['index']] + coupling.orbital2
                spin_orbit_term = np.einsum('c, nm -> cnm', k_dependence(coupling, kpoints),
                                            coupling.get_spin_orbit_matrix_elements())
                matrix[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += spin_orbit_term
                matrix[:, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conj(spin_orbit_term.swapaxes(1, 2))

            if model._is_spin_polarized:
                return matrix[:, ::2, ::2]
        return matrix
        matrix = np.zeros((len(kpoints), len(model.structure), len(model.structure)), dtype=complex)

        # construct matrix elements at each k-point
        # for _, kpoint in enumerate(kpoints):
        for coupling in model.get_set_couplings():
            i, j = coupling.site1.properties['index'], coupling.site2.properties['index']

            # get the matrix elements from the couplings
            # A, inner = coupling.get_tightbinding_matrix_elements(kpoint)
            A = k_dependence(coupling, kpoints) * coupling.get_tightbinding_matrix_elements()
            matrix[:, i, j] += A
            matrix[:, j, i] += np.conj(A)

        # add spin degrees of freedom
        if model.check_if_spinful():
            matrix = np.kron(matrix, np.eye(2))

            for coupling in model.get_set_couplings():
                i, j = coupling.site1.properties['index'], coupling.site2.properties['index']
                spin_orbit_term = np.einsum('c, nm -> cnm', k_dependence(coupling, kpoints), coupling.get_spin_orbit_matrix_elements())
                matrix[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += spin_orbit_term
                matrix[:, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conj(spin_orbit_term.swapaxes(1, 2))

            if model._is_spin_polarized:
                return matrix[:, ::2, ::2]
        return matrix

    def solve(self, solver: Callable[[npt.NDArray[np.complex128]], tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Diagonalizes the Hamiltonian using the provided solver."""

        # allocate memory for the output
        E = np.zeros(self.hamiltonian.shape[0:2])  # complex for white alg.
        psi = np.zeros(self.hamiltonian.shape, dtype=complex)

        # diagonalize the Hamiltonian at each k-point
        for _, k in enumerate(self.kpoints.kpoints):
            try:
                E[_], psi[_] = solver(self.hamiltonian[_])
            except:
                s = 'Hamiltonian is not positive-definite at k = (%.3f, %.3f' \
                    ', %.3f). Adding small epsilon and trying again.' % tuple(k)
                print(s)
                try:
                    epsilon = np.sort(np.real(eigvals(self.hamiltonian[_]))) + 0.0000001
                    # epsilon = 0.1
                    H_shftd = self.hamiltonian[_] + np.eye(self.hamiltonian.shape[1]) * epsilon
                    E[_], psi[_] = solver(H_shftd)
                except:
                    s = 'Diagonalization failed! Check classical ground state' \
                        ' or try different method for approximate' \
                        'diagonalization.'
                    raise TypeError(s)
        return E, psi

    def get_berry_phase(self,
                        band_indices: IntVector = None,
                        energy: float = None) -> float:
        """Computes the Berry phase.

        See Also
        --------
        :class:`topwave.topology.get_berry_phase`

        """

        loop_operator = get_fermionic_wilson_loop(self, band_indices, energy)
        return get_berry_phase(loop_operator)

    def get_particle_density(self,
                             filling: float) -> np.ndarray:
        """Computes the electron density for all occupied states.

        Parameters
        ----------
        filling: float
            The energy up to which states are considered.

        Returns
        -------
        np.ndarray
            Array with the electron densities. The shape is
            (num_unit_cell_x, num_unit_cell_y, num_unit_cell_z, num_sublattices, num_spins)

        """

        wavefunctions = self.psi.transpose(0, 2, 1)
        wavefunctions[self.energies > filling] = 0
        densities = np.real(np.conj(wavefunctions) * wavefunctions).sum(axis=0).sum(axis=0)

        supercell_shape = () if self.model.scaling_factors is None else self.model.scaling_factors
        is_spinless = not self.model.check_if_spinful() or self.model._is_spin_polarized
        num_spins = 1 if is_spinless else 2
        densities = densities.reshape((*supercell_shape,
                                       -1,
                                       num_spins))
        return densities

    def get_spectral_density(self,
                             energy: float = 0.0,
                             broadening: float = 0.02) -> RealList:
        """Computes the spectral density at a given energy for all k-points of the spectrum:

        Parameters
        ----------
        energy: float
            The energy at which the spectral density is computed.
        broadening: float
            Broadening of the spectral density. Default is 0.02.

        Returns
        -------
        RealList
            The spectral density at each k-point of the spectrum.

        Examples
        --------
        Compute the spectral density of something.


        """

        return -np.imag(np.reciprocal(energy - self.energies + 1j * broadening)).sum(axis=1)

