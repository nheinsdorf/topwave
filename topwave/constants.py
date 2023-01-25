"""Collection of physical constants."""
from numpy import array


G_LANDE = 2  # different Lande factors not implemented yet
K_BOLTZMANN = 0.086173324  # given in meV/K
MU_BOHR = 0.057883818066000  # given in meV/Tesla

# Pauli matrices
PAULI_X = array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = array([[1, 0], [0, -1]], dtype=complex)
PAULI_VEC = array([PAULI_X, PAULI_Y, PAULI_Z], dtype=complex)
