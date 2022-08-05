import numpy as np
from numpy.linalg import norm

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
        return np.reciprocal(np.exp(energies / (Model.kB * temperature)) - 1)

def gaussian(x, mean, std):
    """Evaluates the Gauss distribution at x.

    Parameters
    ----------
    x : float or numpy.ndarray
        The values at which the Gaussian distribution should be evaluated.
    mean : float
        The mean value of the distribution.
    std : float
        The standard deviation of the distribution.

    """

    x = np.array([x], dtype=float).flatten()
    return np.exp(-0.5 * np.square((x - mean) / std)) / (std * np.sqrt(2 * np.pi))


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



