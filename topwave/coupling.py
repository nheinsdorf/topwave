from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite

from topwave.types import ComplexVector, Complex2x2, IntVector, Real3x3, Vector
from topwave.util import format_input_vector, pauli

__all__ = ["Coupling"]

# TODO: implement the set_coupling, set_spin_orbit, etc as a function here as well.
@dataclass(slots=True)
class Coupling:
    """Coupling of two sites.

    Examples
    --------


    """

    index: int
    lattice_vector: IntVector
    site1: PeriodicSite
    orbital1: int
    site2: PeriodicSite
    orbital2: int
    symmetry_id: int
    symmetry_op: SymmOp
    distance: float = field(init=False)
    sublattice_vector: Vector = field(init=False)
    is_onsite: bool = False
    is_set: bool = False
    matrix: Real3x3 = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    spin_orbit: Vector = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    strength: complex = 0j

    def __post_init__(self) -> None:
        self.distance = self.site1.distance(self.site2, self.lattice_vector)
        self.is_onsite = True if (self.site1 == self.site2 and (self.lattice_vector == 0).all()) else False
        self.sublattice_vector = self.site2.frac_coords - self.site1.frac_coords

    def get_energy(self) -> float:
        """Returns the classical energy of the coupling."""

        exchange_matrix = self.get_exchange_matrix()
        spin1, spin2 = self.site1.properties['magmom'], self.site2.properties['magmom']

        return spin1 @ exchange_matrix @ spin2

    def get_exchange_matrix(self) -> Real3x3:
        """Returns the exchange matrix."""

        heisenberg_exchange = np.diag([self.strength] * 3)
        antisymmetric_exchange = np.array([[0, self.spin_orbit[2], -self.spin_orbit[1]],
                                           [-self.spin_orbit[2], 0, self.spin_orbit[0]],
                                           [self.spin_orbit[1], -self.spin_orbit[0], 0]], dtype=np.float64)
        general_exchange = self.matrix
        return heisenberg_exchange + antisymmetric_exchange + general_exchange

    def get_fourier_coefficients(
            self,
            kpoint: Vector) -> tuple[complex, ComplexVector]:
        """Returns the Fourier coefficient and its inner derivatives at k_point.

        Parameters
        ----------
        kpoint : np.ndarray((3,), np.float64)
            Input k_point

        Returns
        -------
        Tuple[complex, np.ndarray[(3,), np.complex128)]

        """

        return np.exp(-2j * np.pi * np.dot(self.lattice_vector, kpoint)), -2j * np.pi * self.lattice_vector

    def get_fourier_derivative(self,
                               kpoint: Vector,
                               direction: str) -> complex:
        """Returns the derivative of the Fourier coefficient with respect to one of the crystal momenta.


        Parameters
        ----------
        kpoint: Vector
            The k-point at which the derivative is evaluated.
        direction: str
            Which crystal momentum is used for the derivative. Options are 'x', 'y' and 'z'.

        Returns
        -------
        complex
            The derivative of the couplings Fourier coefficient.

        """

        index = 'xyz'.find(direction)
        coefficient, _ = self.get_fourier_coefficients(kpoint)
        inner_derivative = -2j * np.pi * self.lattice_vector[index]
        return inner_derivative * coefficient

    def get_spinwave_matrix_elements(self) -> tuple[complex, complex, complex, complex, complex, complex, ComplexVector]:
        """Constructs the matrix elements for the Spinwave Hamiltonian at a given k-point."""

        # the local spin reference frames
        rotation1 = self.site1.properties['Rot']
        rotation2 = self.site2.properties['Rot']
        rotation_vector1 = rotation1[:, 0] + 1j * rotation1[:, 1]
        rotation_vector2 = rotation2[:, 0] + 1j * rotation2[:, 1]
        spin_direction1 = rotation1[:, 2]
        spin_direction2 = rotation2[:, 2]

        # total spin prefactor
        spin_magnitude1 = norm(self.site1.properties['magmom'])
        spin_magnitude2 = norm(self.site2.properties['magmom'])
        pre_factor = np.sqrt(spin_magnitude1 * spin_magnitude2) / 2

        # spin exchange matrix
        exchange_matrix = self.get_exchange_matrix()

        # construct k-dependent elements of the diagonal blocks
        A = pre_factor * (rotation_vector1 @ exchange_matrix @ np.conj(rotation_vector2))
        Abar = pre_factor * (rotation_vector1 @ exchange_matrix @ np.conj(rotation_vector2))

        # construct the k-independent elements of the diagonal blocks
        CI = spin_magnitude1 * (spin_direction1 @ exchange_matrix @ spin_direction2)
        CJ = spin_magnitude2 * (spin_direction1 @ exchange_matrix @ spin_direction2)

        # construct the k-dependent elements of the off-diagonal blocks
        B = pre_factor * (rotation_vector1 @ exchange_matrix @ rotation_vector2)
        Bbar = pre_factor * np.conj(rotation_vector2 @ exchange_matrix @ rotation_vector1)

        return A, Abar, CI, CJ, B, Bbar

    def get_tightbinding_matrix_elements(self) -> complex:
        """Constructs the matrix elements for the TightBinding Hamiltonian at a given k-point."""

        # c_k, inner = self.get_fourier_coefficients(k_point)
        # matrix_element = c_k * self.strength
        # return matrix_element, inner
        return self.strength

    def get_spin_orbit_matrix_elements(self) -> Complex2x2:
        """Creates the matrix elements of the tight-binding Hamiltonian that come from spin-orbit interation."""

        # c_k, inner = self.get_fourier_coefficients(k_point)
        # spin_orbit_term = 1j * c_k * pauli(self.spin_orbit, normalize=False)
        spin_orbit_term = 1j * pauli(self.spin_orbit, normalize=False)
        return spin_orbit_term

    def set_matrix(self, matrix: Real3x3):
        """Sets an interaction matrix.

        """

        self.matrix = np.array(matrix, dtype=np.float).reshape((3, 3))
        self.is_set = True

    # NOTE: should I make this a property? Check how it works with dataclasses.
    def set_coupling(self, strength: float, overwrite: bool = True) -> None:
        """Sets the couplings."""

        new_strength = strength if overwrite else strength + self.strength
        self.strength = new_strength
        self.is_set = True

    def set_spin_orbit(self, input_vector: Vector, strength: float = None) -> None:
        """Sets the spin-orbit coupling."""

        input_vector = format_input_vector(orientation=input_vector, length=strength)
        self.spin_orbit = input_vector
        self.is_set = True

    def unset(self) -> None:
        """Sets strength and spin_orbit coupling to zero and unsets itself."""

        self.strength = 0.
        self.spin_orbit = np.zeros(3, dtype=np.float64)
        self.is_set = False
