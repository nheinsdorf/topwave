from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numpy import linalg
import numpy.typing as npt

from topwave.constants import K_BOLTZMANN
if TYPE_CHECKING:
    from topwave.model import Model


def bose_distribution(energies, temperature):
    """Calculates the Bose-Einstein distribution for a given set of energies and a temperature.

    Parameters
    ----------
    energies : float or numpy.ndarray
        Energies for which the distribution is evaluated.
    temperature : float
        The temperature given in Kelvin.

    Returns
    -------
    A numpy.ndarray with the Bose-Einstein distribution for each energy.

    """

    energies = np.array([energies], dtype=float).flatten()
    # NOTE: build in a check for zero energy?!
    if temperature == 0:
        return np.zeros(energies.shape, dtype=float)
    else:
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


def get_boundary_couplings(model: Model, direction: str = 'xyz') -> npt.NDArray[np.int64]:
    """Returns indices of couplings that change the unit cell in a given direction."""

    x_indices = [coupling.index for coupling in model.couplings if coupling.lattice_vector[0] != 0] if 'x' in direction else []
    y_indices = [coupling.index for coupling in model.couplings if coupling.lattice_vector[1] != 0] if 'y' in direction else []
    z_indices = [coupling.index for coupling in model.couplings if coupling.lattice_vector[2] != 0] if 'z' in direction else []
    return np.unique(np.concatenate((x_indices, y_indices, z_indices), axis=0)).astype(np.int64)


def format_input_vector(orientation: list[float] | npt.NDArray[np.float64], length: float = None) -> npt.NDArray[np.float64]:
    """Normalizes an input vector and scales it by length, or does nothing if length=None."""

    unscaled_vector = np.array(orientation, dtype=float).reshape((3,))
    out = unscaled_vector if length is None else length * unscaled_vector / linalg.norm(unscaled_vector)
    return out


def gaussian(x, mean, std, normalize=True):
    """Evaluates the normal distribution at x.

    Parameters
    ----------
    x : float or numpy.ndarray
        The values at which the Gaussian distribution should be evaluated.
    mean : float
        The mean value of the distribution.
    std : float
        The standard deviation of the distribution.
    normalize : bool
        If true, the integral over the distribution will be normalized to one. If false, the pre-factor
        1 / (sqrt(2pi) * sigma) is omitted. Default is true.

    """

    x = np.array([x], dtype=float).flatten()
    pre_factor = 1 / (std * np.sqrt(2 * np.pi)) if normalize else 1
    return pre_factor * np.exp(-0.5 * np.square((x - mean) / std))


def get_azimuthal_angle(vector, deg=False):
    """Returns the azimuthal angle of a three component vector w.r.t. [1, 0, 0].

    Parameters
    ----------
    vector : list or numpy.ndarray
        Three-dimensional input vector.
    deg : bool
        If True, the output is given in degrees. Default is False.

    Returns
    -------
    angle : float
        The azimuthal angle of the input vector in radians.

    """

    vector = np.array(vector, dtype=float).reshape((3,))
    vector /= linalg.norm(vector)
    angle = np.arccos(vector @ [1, 0, 0])
    if deg:
        return np.rad2deg(angle)
    else:
        return angle


def get_elevation_angle(vector, deg=False):
    """Returns the elevation angle of a three component vector w.r.t. [0, 0, 1].

    Parameters
    ----------
    vector : list or numpy.ndarray
        Three-dimensional input vector.
    deg : bool
        If True, the output is given in degrees. Default is False.

    Returns
    -------
    angle : float
        The elevation angle of the input vector in radians.

    """

    vector = np.array(vector, dtype=float).reshape((3,))
    vector /= linalg.norm(vector)
    angle = np.arccos(vector @ [0, 0, 1])
    if deg:
        return np.rad2deg(angle)
    else:
        return angle


class Pauli:
    """Class that holds the Pauli matrices.

    Parameters
    ----------
    d : list
        Three-dimensional list that is used to create linear combination of the Pauli matrices
        with d as coefficients.
    normalize : bool
        If true, d is normalized. Default is true.

    Returns
    -------
    Returns the (normalized) linear combination of Pauli matrices. It does NOT return a class instance.

    """
    id = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    vec = np.array([x, y, z], dtype=complex)

    def __init__(self):
        pass
    def __new__(cls, d, normalize=True):
        d = np.array(d).reshape((3,))
        if normalize:
            d = d / linalg.norm(d)
        return np.einsum('i, inm -> nm', d, cls.vec)


def rotate_vector(input, angle, rotation_axis, basis=None):
    """Rotates a 3 component vector by a given angle (in radians) about an arbitrary axis.

    Parameters
    ----------
    input : list or numpy.ndarray
        Three-dimensional input vector.
    angle : float
        Angle given in radians.
    rotation_axis : list or numpy.ndarray
        Three dimensional vector specifying the rotation axis.
    basis : numpy.ndarray
        Three-by-three array that indicates the basis in which the rotation matrix is transformed.
        If None, cartesian coordinates are used. Default is None.

    Returns
    -------
    output : numpy.ndarray
        The rotated input vector.

    """

    input = np.array(input, dtype=float).reshape((3,))
    rotation_axis = np.array(rotation_axis, dtype=float).reshape((3,)) / linalg.norm(rotation_axis)

    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]], dtype=float)
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]], dtype=float)
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]], dtype=float)

    rotation = np.einsum('n, nij -> ij', rotation_axis, [rotation_x, rotation_y, rotation_z])

    basis = np.eye(3) if basis is None else np.array(basis, dtype=float).reshape((3, 3))

    return linalg.inv(basis) @ rotation @ basis @ input


def rotate_vector_to_ez(v):
    """Creates a 3x3 rotation matrix R with R v = [0, 0, 1]

            Parameters
            ----------
            v : numpy.ndarray
                Three-dimensional vector.

            Returns
            -------
            numpy.ndarray
                3x3 rotation matrix R with R v = [0, 0, 1].

            """

    v = np.array(v, dtype=float) / linalg.norm(v)
    e3 = v
    if np.isclose(np.abs(v), [1, 0, 0], atol=0.00001).all():
        e2 = np.array([0, 0, 1])
    else:
        e2 = np.cross(v, [1, 0, 0])
    e2 = e2 / linalg.norm(e2)
    e1 = np.cross(e2, e3)

    return np.array([e1, e2, e3]).T
