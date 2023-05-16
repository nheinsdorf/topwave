from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy.linalg import eigh, eigvals, inv, multi_dot

from topwave import solvers
from topwave.constants import G_LANDE, MU_BOHR
from topwave.set_of_kpoints import SetOfKPoints
from topwave.model import Model, SpinWaveModel, TightBindingModel
from topwave.util import pauli

__all__ = ["Spec"]

# CHECK for model.dim
@dataclass(slots=True)
class Spec:
    """Computes the spectrum of a model for a given set of k-points."""

    model: Model
    k_points: list[list[float]] | npt.NDArray[np.float64]
    energies: npt.NDArray[np.float64] = field(init=False)
    hamiltonian: npt.NDArray[np.float64] = field(init=False)
    k_points_xyz: npt.NDArray[np.float64] = field(init=False)
    psi: npt.NDArray[np.float64] = field(init=False)

    # TODO: refactor k_points to kpoints
    def __post_init__(self) -> None:
        if isinstance(self.k_points, SetOfKPoints):
            self.k_points = self.k_points.kpoints
        else:
            self.k_points = np.array(self.k_points, dtype=np.float64).reshape((-1, 3))

        self.k_points_xyz = 2 * np.pi * np.einsum('ka, ab -> kb', self.k_points, inv(self.model.structure.lattice.matrix))

        if self.model.type == 'spinwave':
            constructor, solver = self.get_spinwave_hamiltonian, solvers.colpa
        else:
            constructor, solver = self.get_tightbinding_hamiltonian, eigh

        self.hamiltonian = constructor(self.model, self.k_points)
        self.energies, self.psi = self.solve(solver)

    def get_spinwave_hamiltonian(self, model: SpinWaveModel, k_points: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Constructs the spin wave Hamiltonian for a set of given k-points."""

        dim = len(self.model.structure)
        matrix = np.zeros((len(k_points), 2 * dim, 2 * dim), dtype=complex)

        # construct matrix elements at each k-point
        for _, k_point in enumerate(k_points):
            for coupling in model.get_set_couplings():

                i, j = coupling.site1.properties['index'], coupling.site2.properties['index']

                # get the matrix elements from the couplings
                (A, Abar, CI, CJ, B12, B21, inner) = coupling.get_spinwave_matrix_elements(k_point)

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
            u = self.model.structure[_].properties['Rot'][:, 0] + 1j * self.model.structure[_].properties['Rot'][:, 1]
            K = np.diag(self.model.structure[_].properties['onsite_vector'])
            matrix[:, _, _] += u @ K @ np.conj(u)
            matrix[:, _ + dim, _ + dim] += np.conj(u @ K @ np.conj(u))
            matrix[:, _, _ + dim] += u @ K @ u
            matrix[:, _ + dim, _] += np.conj(u @ K @ u)

        # add the external magnetic field
        for _ in range(dim):
            v = self.model.structure[_].properties['Rot'][:, 2]
            H_Zeeman = MU_BOHR * G_LANDE * np.dot(model.zeeman, v)
            matrix[:, _, _] += H_Zeeman
            matrix[:, _ + dim, _ + dim] += H_Zeeman

        return matrix

    def get_tightbinding_hamiltonian(self, model: TightBindingModel, k_points: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Constructs the spin wave Hamiltonian for a set of given k-points."""

        matrix = np.zeros((len(k_points), len(self.model.structure), len(self.model.structure)), dtype=complex)

        # construct matrix elements at each k-point
        for _, k_point in enumerate(k_points):
            for coupling in model.get_set_couplings():
                i, j = coupling.site1.properties['index'], coupling.site2.properties['index']

                # get the matrix elements from the couplings
                A, inner = coupling.get_tightbinding_matrix_elements(k_point)

                matrix[_, i, j] += A
                matrix[_, j, i] += np.conj(A)

        for _, site in enumerate(self.model.structure):
            matrix[:, _, _] += site.properties['onsite_scalar']

        # add spin degrees of freedom
        if model.check_if_spinful():
            matrix = np.kron(matrix, np.eye(2))

            for _, site in enumerate(self.model.structure):
                matrix[:, 2 * _: 2 * _ + 2, 2 * _: 2 * _ + 2] += MU_BOHR * G_LANDE * pauli(model.zeeman, normalize=False)

                # add onsite term
                matrix[:, 2 * _: 2 * _ + 2, 2 * _: 2 * _ + 2] += pauli(site.properties['onsite_vector'], normalize=False)

            # add spin-orbit term
            for _, k_point in enumerate(k_points):
                for coupling in model.get_set_couplings():
                    i, j = coupling.site1.properties['index'], coupling.site2.properties['index']
                    spin_orbit_term, inner = coupling.get_spin_orbit_matrix_elements(k_point)
                    matrix[_, 2 * i:2 * i + 2, 2 * j:2 * j + 2] += spin_orbit_term
                    matrix[_, 2 * j:2 * j + 2, 2 * i:2 * i + 2] += np.conj(spin_orbit_term.T)

        return matrix

    def solve(self, solver: Callable[[npt.NDArray[np.complex128]], tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Diagonalizes the Hamiltonian using the provided solver."""

        # allocate memory for the output
        E = np.zeros(self.hamiltonian.shape[0:2])  # complex for white alg.
        psi = np.zeros(self.hamiltonian.shape, dtype=complex)

        # diagonalize the Hamiltonian at each k-point
        for _, k in enumerate(self.k_points):
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


    def get_wilson_loop_operator(self, occ):
        """Returns the fermionic wilson loop operator.

        NOTE: Make the distinction by the Type of the spectrum.

        Parameters
        ----------
        occ : list
            List of occupied bands.

        """

        # select the wavefunctions of occupied bands
        psi_right = self.psi[:, :, occ]

        # check whether start and end k-point are the same and impose closed loop
        if np.all(np.isclose(self.k_points[0], self.k_points[-1])):
            psi_right[0] = psi_right[-1]
        else:
            # implement the case where they are connected by a reciprocal vector
            # https://github.com/bellomia/PythTB/blob/master/pythtb.py
            # see 'impose_pbc'-method
            pass

        # construct bra-eigenvectors for k+1
        psi_left = np.roll(np.conj(psi_right), 1, axis=0)

        # compute num_k - 1 overlaps
        F = np.einsum('knm, knl -> kml', psi_left[1:], psi_right[1:])

        # take the product of the matrices to compute the wilson loop operator
        self.wilson_loop_operator = multi_dot(F)
