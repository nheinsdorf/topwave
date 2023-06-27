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
from topwave.util import format_kpoints, pauli

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

        .. math:: \rho(\omega) = \int_{k} dk A_k(\omega).


        .. admonition:: K-Grid!
            :class: warning

            The spectral density should be evaluated on a grid that covers the whole Brillouin zone. You can use
            :class:`topwave.set_of_kpoints.Plane` or grid.

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

        dim = len(model.structure)
        matrix = np.zeros((len(kpoints), 2 * dim, 2 * dim), dtype=complex)

        # construct matrix elements at each k-point
        for _, kpoint in enumerate(kpoints):
            for coupling in model.get_set_couplings():

                i, j = coupling.site1.properties['index'], coupling.site2.properties['index']

                # get the matrix elements from the couplings
                (A, Abar, CI, CJ, B12, B21, inner) = coupling.get_spinwave_matrix_elements(kpoint)

                matrix[_, i, j] += A
                matrix[_, j, i] += np.conj(A)
                matrix[_, i + dim, j + dim] += np.conj(Abar)
                matrix[_, j + dim, i + dim] += Abar

                matrix[_, i, i] -= CI
                matrix[_, j, j] -= CJ
                matrix[_, i + dim, i + dim] -= np.conj(CI)
                matrix[_, j + dim, j + dim] -= np.conj(CJ)

                matrix[_, i, j + dim] += B12
                matrix[_, j, i + dim] += B21
                matrix[_, j + dim, i] += np.conj(B12)
                matrix[_, i + dim, j] += np.conj(B21)

        # add single ion anisotropies
        for _ in range(dim):
            u = model.structure[_].properties['Rot'][:, 0] + 1j * model.structure[_].properties['Rot'][:, 1]
            K = np.diag(model.structure[_].properties['onsite_vector'])
            matrix[:, _, _] += u @ K @ np.conj(u)
            matrix[:, _ + dim, _ + dim] += np.conj(u @ K @ np.conj(u))
            matrix[:, _, _ + dim] += u @ K @ u
            matrix[:, _ + dim, _] += np.conj(u @ K @ u)

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
            kpoints: VectorList | SetOfKPoints,
            derivative: str = None) -> SquareMatrix:
        """Constructs the Tight Binding Hamiltonian (or its derivative) for a set of given k-points.


        Parameters
        ----------
        model: TightBindingModel
            The model of that is used to calculate the spectrum.
        kpoints: VectorList | SetOfKPoints
            The kpoints at which the Hamiltonian or its derivatives are constructed.
        derivative: str
            String that indicates which derivative should be constructed. Options are 'x', 'y' or 'z'.
            Default is None.

        Returns
        -------
        SquareMatrix
            The Hamiltonian or one of its derivatives.

        """

        kpoints = format_kpoints(kpoints)

        if derivative is None:
            k_dependence = get_periodic_fourier_coefficient
        else:
            k_dependence = partial(get_periodic_fourier_derivative, direction=derivative)
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

        for _, site in enumerate(model.structure):
            matrix[:, _, _] += site.properties['onsite_scalar']

        # add spin degrees of freedom
        if model.check_if_spinful():
            matrix = np.kron(matrix, np.eye(2))

            for _, site in enumerate(model.structure):
                matrix[:, 2 * _: 2 * _ + 2, 2 * _: 2 * _ + 2] += MU_BOHR * G_LANDE * pauli(model.zeeman, normalize=False)

                # add onsite term
                matrix[:, 2 * _: 2 * _ + 2, 2 * _: 2 * _ + 2] += pauli(site.properties['onsite_vector'], normalize=False)

            # add spin-orbit term
            # for _, kpoint in enumerate(kpoints):
            for coupling in model.get_set_couplings():
                i, j = coupling.site1.properties['index'], coupling.site2.properties['index']
                spin_orbit_term = np.einsum('c, nm -> cnm', k_dependence(coupling, kpoints), coupling.get_spin_orbit_matrix_elements())
                matrix[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += spin_orbit_term
                matrix[:, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conj(spin_orbit_term.swapaxes(1, 2))

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


    def get_berry_phase(self, occupied: IntVector) -> float:
        """Computes the Berry phase.

        See Also
        --------
        :class:`topwave.topology.get_berry_phase`

        """

        loop_operator = get_fermionic_wilson_loop(self, occupied)
        return get_berry_phase(loop_operator)

    def get_spectral_density(self,
                             energy: float = 0.0,
                             broadening: float = 0.02) -> RealList:
        """Computes the spectral density at a given energy for all k-points of the spectrum:

        .. math:: A_k(\omega) = -\operatorname{Im} \sum_n \frac{1}{\omega - \epsilon_n + i\eta}.


        .. admonition:: Normalization!
            :class: warning

            The spectral density is only evaluated at one energy and is not normalized.

        asdf

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

