from __future__ import annotations

import numpy as np

from topwave.types import IntVector, VectorList
from topwave.util import get_span_indices

__all__ = ["PlaquetteCover"]

def PlaquetteCover(normal: str,
                   num_x: int = 10,
                   num_y: int = 10,
                   x_min: float = -0.5,
                   x_max: float = 0.5,
                   y_min: float = -0.5,
                   y_max: float = 0.5) -> list[VectorList]:
    """Builds a rectangular cover of a plane through the Brillouin zone.

    Parameters
    ----------
    normal: str
        A string indicating the normal of the plane in units of reciprocal lattice vectors.
        Options are 'x', 'y' or 'z'.
    num_x: int
        Number of plaquettes along the first vector that spans the plane.
    num_y: int
        Number of plaquettes along the second vector that spans the plane.
    x_min: float
        First component of the origin of the cover.
    x_max: float
        First component of the end point of the cover.
    y_min: float
        Second component of the origin of the cover.
    y_max: float
        Second component of the end point of the cover.

    Returns
    -------
    list[VectorList]
        A list of sets of four reciprocal vectors that correspond to the corner points of the plaquettes.

    Notes
    -----
    If the cover is used to compute topological invariants, e.g. the Berry curvature,
    make sure that the cover spans the **whole Brillouin zone** exactly once. In other words,
    do not touch `x_min`, `x_max`, `y_min` and `y_max`.

    Examples
    --------

    We create a 10-by-10 cover of the xy-plane.

    .. ipython:: python

        print('lol')

    """

    x = np.linspace(x_min, x_max, num_x, endpoint=False)
    y = np.linspace(y_min, y_max, num_y, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = np.zeros(xx.shape, dtype=np.float64)


    id_x, id_y, id_z = get_span_indices(normal)

