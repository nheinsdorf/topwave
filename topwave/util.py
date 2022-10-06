import numpy as np
from numpy.linalg import inv, norm

kB = 0.086173324  # given in meV/K


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
        return np.reciprocal(np.exp(energies / (kB * temperature)) - 1)


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

def get_azimuthal_angle(vector):
    """Returns the azimuthal angle of a three component vector w.r.t. [1, 0, 0].

    Parameters
    ----------
    vector : list or numpy.ndarray
        Three-dimensional input vector.

    Returns
    -------
    angle : float
        The azimuthal angle of the input vector in radians.

    """

    vector = np.array(vector, dtype=float).reshape((3,))
    vector = vector / norm(vector)
    return np.arccos(vector @ [1, 0, 0])

def get_elevation_angle(vector):
    """Returns the elevation angle of a three component vector w.r.t. [0, 0, 1].

    Parameters
    ----------
    vector : list or numpy.ndarray
        Three-dimensional input vector.

    Returns
    -------
    angle : float
        The elevation angle of the input vector in radians.

    """

    vector = np.array(vector, dtype=float).reshape((3,))
    vector = vector / norm(vector)
    return np.arccos(vector @ [0, 0, 1])

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
            d = d / norm(d)
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
    rotation_axis = np.array(rotation_axis, dtype=float).reshape((3,)) / norm(rotation_axis)

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

    return inv(basis) @ rotation @ basis @ input

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

    v = np.array(v, dtype=float) / norm(v)
    e3 = v
    if np.isclose(np.abs(v), [1, 0, 0], atol=0.00001).all():
        e2 = np.array([0, 0, 1])
    else:
        e2 = np.cross(v, [1, 0, 0])
    e2 = e2 / norm(e2)
    e1 = np.cross(e2, e3)

    return np.array([e1, e2, e3]).T
