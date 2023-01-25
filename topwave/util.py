from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy import linalg
import numpy.typing as npt

from topwave.constants import K_BOLTZMANN, PAULI_VEC
if TYPE_CHECKING:
    from topwave.model import Model


def bose_distribution(energies: float | npt.ArrayLike, temperature: float) -> npt.NDArray[np.float64]:
    """Calculates the Bose-Einstein distribution for a given set of energies and a temperature."""

    energies = np.array([energies], dtype=float).flatten()
    if temperature == 0:
        return np.zeros(energies.shape, dtype=float)
    return np.reciprocal(np.exp(energies / (K_BOLTZMANN * temperature)) - 1)


def coupling_selector(attribute: str, value: int | float, model: Model) -> list[int]:
    """Selects a couplings based on a given attribute."""

    match attribute:
        case 'is_set':
            indices = [coupling.index for coupling in model.couplings if coupling.is_set == value]
        case 'index':
            indices = [coupling.index for coupling in model.couplings if coupling.index == value]
        case 'symmetry_id':
            indices = [coupling.index for coupling in model.couplings if coupling.symmetry_id == value]
        case 'distance':
            indices = [coupling.index for coupling in model.couplings if np.isclose(coupling.distance, value, atol=1e-5)]
    return indices


def format_input_vector(orientation: list[float] | npt.NDArray[np.float64], length: Optional[float] = None) -> npt.NDArray[np.float64]:
    """Normalizes an input vector and scales it by length, or does nothing if length=None."""

    unscaled_vector = np.array(orientation, dtype=float).reshape((3,))
    out = unscaled_vector if length is None else length * unscaled_vector / linalg.norm(unscaled_vector)
    return out


def gaussian(x: int | float | npt.NDArray[np.float64], mean: float, std: float, normalize: bool = True) -> npt.NDArray[np.float64]:
    """Evaluates the normal distribution at x."""

    x = np.array([x], dtype=float).flatten()
    pre_factor = 1 / (std * np.sqrt(2 * np.pi)) if normalize else 1
    return pre_factor * np.exp(-0.5 * np.square((x - mean) / std))


def get_azimuthal_angle(vector: npt.ArrayLike, deg: bool = False) -> float:
    """Returns the azimuthal angle of a three component vector w.r.t. [1, 0, 0]."""

    vector = np.array(vector, dtype=float).reshape((3,))
    vector /= linalg.norm(vector)
    angle = np.arccos(vector @ [1, 0, 0])
    if deg:
        return np.rad2deg(angle)
    return angle


def get_boundary_couplings(model: Model, direction: str = 'xyz') -> npt.NDArray[np.int64]:
    """Returns indices of couplings that change the unit cell in a given direction."""

    x_indices = [coupling.index for coupling in model.couplings if coupling.lattice_vector[0] != 0] if 'x' in direction else []
    y_indices = [coupling.index for coupling in model.couplings if coupling.lattice_vector[1] != 0] if 'y' in direction else []
    z_indices = [coupling.index for coupling in model.couplings if coupling.lattice_vector[2] != 0] if 'z' in direction else []
    return np.unique(np.concatenate((x_indices, y_indices, z_indices), axis=0)).astype(np.int64)


def get_elevation_angle(vector: npt.ArrayLike, deg: bool = False) -> float:
    """Returns the elevation angle of a three component vector w.r.t. [0, 0, 1]."""

    vector = np.array(vector, dtype=float).reshape((3,))
    vector /= linalg.norm(vector)
    angle = np.arccos(vector @ [0, 0, 1])
    if deg:
        return np.rad2deg(angle)
    return angle


def pauli(vector: npt.ArrayLike, normalize: bool = True) -> npt.NDArray[np.complex128]:
    """Outputs a linear combination of Pauli matrices given by the input vector."""

    vector = np.array(vector).reshape((3,))
    if normalize:
        vector = vector / linalg.norm(vector)
    return np.einsum('i, inm -> nm', vector, PAULI_VEC)


def rotate_vector(vector: npt.ArrayLike, angle: float, axis: npt.ArrayLike,
                  basis: Optional[npt.ArrayLike] = None) -> npt.NDArray[np.float64]:
    """Rotates a 3 component vector by a given angle (in radians) about an arbitrary axis."""

    vector = np.array(vector, dtype=float).reshape((3,))
    axis = np.array(axis, dtype=float).reshape((3,)) / linalg.norm(axis)

    rotation = np.array([[np.cos(angle) + (1 - np.cos(angle)) * axis[0]**2,
                          axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                          axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                         [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                          np.cos(angle) + (1 - np.cos(angle)) * axis[1]**2,
                          axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                         [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                          axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                          np.cos(angle) + (1 - np.cos(angle)) * axis[2]**2]], dtype=np.float64)

    basis = np.eye(3) if basis is None else np.array(basis, dtype=float).reshape((3, 3))

    return linalg.inv(basis) @ rotation @ basis @ vector


def rotate_vector_to_ez(vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Creates a 3x3 rotation matrix R with R v = [0, 0, 1]."""

    vector = np.array(vector, dtype=float).reshape((3,)) / linalg.norm(vector)
    column3 = vector
    if np.isclose(np.abs(vector), [1, 0, 0], atol=0.00001).all():
        column2 = np.array([0, 0, 1])
    else:
        column2 = np.cross(vector, [1, 0, 0])
    column2 = column2 / linalg.norm(column2)
    column1 = np.cross(column2, column3)

    return np.array([column1, column2, column3]).T
