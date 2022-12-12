#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:32:35 2022

@author: niclas
"""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite

from topwave.util import Pauli


@dataclass(slots=True)
class Coupling:
    """Coupling of two sites."""

    index: int
    lattice_vector: npt.NDArray[np.int64]
    site1: PeriodicSite
    site2: PeriodicSite
    symmetry_id: int
    symmetry_op: SymmOp
    distance: float = field(init=False)
    site1_id: int = field(init=False)
    sublattice_vector: npt.NDArray[np.float64] = field(init=False)
    site2_id: int = field(init=False)
    is_set: bool = False
    spin_orbit: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    strength: float = 0.

    def __post_init__(self) -> None:
        self.site1_id = self.site1.properties['id']
        self.site2_id = self.site2.properties['id']
        self.sublattice_vector = self.site2.frac_coords - self.site1.frac_coords

    def get_energy(self) -> float:
        """Returns the classical energy of the coupling."""

        exchange_matrix = self.get_exchange_matrix()
        spin1, spin2 = self.site1.properties['magmom'], self.site2.properties['magmom']

        return spin1 @ exchange_matrix @ spin2

    def get_exchange_matrix(self) -> np.ndarray:
        """Returns the exchange matrix."""

        return np.array([[self.strength, self.spin_orbit[2], -self.spin_orbit[1]],
                         [-self.spin_orbit[2], self.strength, self.spin_orbit[0]],
                         [self.spin_orbit[1], -self.spin_orbit[0], self.strength]], dtype=complex)

    def get_fourier_coefficients(
            self,
            k: npt.ArrayLike) -> tuple[complex, npt.NDArray[np.complex128]]:
        """Returns the Fourier coefficient and its inner derivatives at k."""

        return np.exp(-2j * np.pi * np.dot(self.lattice_vector, k)), -2j * np.pi * self.lattice_vector

    def get_sw_matrix_elements(
            self,
            k: npt.ArrayLike) -> tuple[complex, complex, complex, complex, complex, complex, np.ndarray]:
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

        # phase factors and their derivatives
        phase_factor, inner = self.get_fourier_coefficients(k)

        # spin exchange matrix
        exchange_matrix = self.get_exchange_matrix()

        # construct k-dependent elements of the diagonal blocks
        A = pre_factor * phase_factor * (rotation_vector1 @ exchange_matrix @ np.conj(rotation_vector2))
        Abar = pre_factor * np.conj(phase_factor) * (rotation_vector1 @ exchange_matrix @ np.conj(rotation_vector2))

        # construct the k-independent elements of the diagonal blocks
        CI = spin_magnitude1 * (spin_direction1 @ exchange_matrix @ spin_direction2)
        CJ = spin_magnitude2 * (spin_direction1 @ exchange_matrix @ spin_direction2)

        # construct the k-dependent elements of the off-diagonal blocks
        B = pre_factor * phase_factor * (rotation_vector1 @ exchange_matrix @ rotation_vector2)
        Bbar = pre_factor * np.conj(phase_factor) * np.conj(rotation_vector2 @ exchange_matrix @ rotation_vector1)

        return A, Abar, CI, CJ, B, Bbar, inner

    def get_tb_matrix_elements(self, k: npt.ArrayLike) -> tuple[complex, np.ndarray]:
        """Constructs the matrix elements for the TightBinding Hamiltonian at a given k-point."""

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
