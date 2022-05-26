import numpy as np
from numpy.linalg import norm

def rotate_vector_to_ez(v):
    """ Creates a 3x3 rotation matrix R with R v = [0, 0, 1]

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
