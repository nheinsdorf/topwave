from __future__ import annotations

import numpy as np
from skimage.util import view_as_windows

from topwave.set_of_kpoints import Plane
from topwave.types import VectorList
from topwave.util import get_plaquette_indices


# __all__ = ["get_plaquette_cover"]
#NOTE: make normal also a vector like in set_of_kpoints plane object
def get_line_cover(normal: str,
                   direction: str,
                   num_lines: int,
                   num_points: int,
                   anchor: float = 0.0,
                   min: float = -0.5,
                   max: float = 0.5) -> list[VectorList]:
    """Builds a rectangular cover of a plane through the Brillouin zone.

    Parameters
    ----------
    normal: str
        A string indicating the normal of the plane in units of reciprocal lattice vectors.
        Options are 'x', 'y' or 'z'.
    direction: str
        In which direction the lines are stacked. Options are 'x' and 'y'.
    num_lines: int
        Number of lines that are used to span the plane.
    num_points: int
        Number of points on one line.
    anchor: float
        Where along the normal the plane is anchored. Default is 0.
    min: float
        At which coordinate along the direction the first line is anchored.
    max: float
        At which coordinate along the direction the last line is anchored.

    Returns
    -------
    list[VectorList]
        A num_lines-long list of lines that each contains num_points k-points. The first and last k-point are
        connected by a reciprocal lattice vector.

    Examples
    --------

    We create a cover of the xy-plane anchored at z = 0 with 30 lines stacked along the x-direction.
    Each line consists fo 60 points.

    .. ipython:: python

        import numpy as np
        import matplotlib.pyplot as plt

        # Create the cover.
        cover = tp.get_line_cover('z', 'x', 30, 60)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for line in cover:
            ax.plot(*line.T, c='hotpink')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel(r'$k_x$');
        ax.set_ylabel(r'$k_y$');
        @savefig line_cover.png
        ax.set_zlabel(r'$k_z$');

    """

    normal_vector = [0, 0, 0]
    normal_vector['xyz'.find(normal)] = 1
    if direction == 'x':
        kpoints = Plane(normal_vector, num_lines, num_points, anchor=anchor, x_min=min, x_max=max, endpoint_y=True).kpoints
        return kpoints.reshape((num_lines, num_points, 3))
    kpoints = Plane(normal_vector, num_points, num_lines, anchor=anchor, y_min=min, y_max=max, endpoint_x=True).kpoints
    return kpoints.reshape((num_points, num_lines, 3)).transpose(1, 0, 2)


def get_plaquette_cover(normal: str,
                        num_x: int,
                        num_y: int,
                        anchor: float = 0.0,
                        x_min: float = -0.5,
                        x_max: float = 0.5,
                        y_min: float = -0.5,
                        y_max: float = 0.5,
                        closed: bool = True) -> list[VectorList]:
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
    anchor: float
        Where along the normal the plane is anchored. Default is 0.
    x_min: float
        First component of the first point of the cover.
    x_max: float
        First component of the end point of the cover.
    y_min: float
        Second component of the first point of the cover.
    y_max: float
        Second component of the end point of the cover.
    closed: bool
        If True, the first and last k-point of the plaquette are identified with each other by adding a fifth point,
        e.g. for the calculation of Wilson loops. Default is True.

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
        cover = tp.get_plaquette_cover('z', 10, 10, closed=False)

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
    plaquettes = plaquettes.reshape((3, num_x * num_y, 4))[:, :, [0, 1, 3, 2]].transpose(1, 2, 0)
    if closed:
        return list(np.concatenate((plaquettes, plaquettes[:, 0, :].reshape((num_x * num_y, 1, 3))), axis=1))
    return list(plaquettes)

# def get_scattering_indices(qpoints: SetOfKPoints | VectorList,
#                            kpoints: SetOfKPoints | VectorList)



