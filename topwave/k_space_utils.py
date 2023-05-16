from __future__ import annotations

import numpy as np
from skimage.util import view_as_windows

from topwave.types import VectorList
from topwave.util import get_plaquette_indices


__all__ = ["get_plaquette_cover"]

def get_plaquette_cover(normal: str,
                        anchor: float = 0.0,
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
    anchor: float
        Where along the normal the plane is anchored. Default is 0.
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

    We create a 10-by-10 cover of the xy-plane anchored at z = 0.

    .. ipython:: python

        import numpy as np
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt

        # Create the cover.
        cover = tp.get_plaquette_cover('z')

        # We plot each plaquette as a polygon and give it a random color.
        np.random.seed(188)

        fig, ax = plt.subplots()
        patches = []
        for plaquette in cover:
            patches.append(Polygon(plaquette[:, :2], closed=True))

        colors = 100 * np.random.rand(len(patches))
        p = PatchCollection(patches, cmap=plt.cm.hsv)
        p.set_array(colors)
        ax.add_collection(p)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        @savefig plaquette_cover.png
        ax.set_aspect('equal')

    """

    x = np.linspace(x_min, x_max, num_x + 1, endpoint=True)
    y = np.linspace(y_min, y_max, num_y + 1, endpoint=True)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = np.zeros((num_x, num_y, 2, 2), dtype=np.float64) + anchor
    id_x, id_y, id_z = get_plaquette_indices(normal)
    plaquettes = np.array([view_as_windows(xx, (2, 2)),
                           view_as_windows(yy, (2, 2)),
                           zz], dtype=np.float64)[[id_x, id_y, id_z]]
    return list(plaquettes.reshape((3, num_x * num_y, 4))[:, :, [0, 1, 3, 2]].transpose(1, 2, 0))




