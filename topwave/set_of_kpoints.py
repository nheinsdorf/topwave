from __future__ import annotations

import numpy as np
from pymatgen.core.structure import Structure

from topwave.types import RealList, Vector, VectorList
from topwave.util import rotate_vector_to_ez

__all__ = ["Circle", "Grid", "Path", "Plane", "SetOfKPoints"]

class SetOfKPoints:
    """Class that holds a number of k-points in three dimensional reciprocal space.

    Parameters
    ----------
    kpoints : VectorList
        List of points in reciprocal space in reduced coordinates.

    Attributes
    ----------
    kpoints : VectorList
        This is where kpoints is stored.
    num_kpoints : int
        Number of kpoints.

    """

    def __init__(self, kpoints: VectorList):
        self.kpoints = np.array(kpoints, dtype=np.float64).reshape((-1, 3))
        self.num_kpoints = len(self.kpoints)


class Circle(SetOfKPoints):
    """A circle through reciprocal space.

    Parameters
    ----------
    radius: float
        Radius in reduced reciprocal lattice units of the circle.
    center: Vector
        Where in reciprocal space the circle is centered.
    normal: Vector
        The normal of the plane the circle lies in.
    num_kpoints: int
        The number of k-points used to parameterize the circle. Default is 100.
    endpoint: bool
        If True, the first and last point are identified, e.g. for Wilson loop calculations.
        Default is True.

    Attributes
    ----------
    kpoints: VectorList
        List of points in reciprocal space in reduced coordinates.
    normal: Vector
        This is where normal is saved.
    num_kpoints : int
        Number of k-points.

    Examples
    --------

    Let's create a path from Gamma to M to K and back to Gamma.

    .. ipython:: python

        circle = tp.Circle(radius=0.2, center=[0, 0, 0], normal=[1, 0, 1])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(*circle.kpoints.T)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        @savefig circle.png
        ax.set_zlabel(r'$k_z$')

    """

    def __init__(self,
                 radius: float,
                 center: Vector,
                 normal: Vector,
                 num_kpoints: int = 100,
                 endpoint: bool = True) -> None:


        angles = np.linspace(0, 2 * np.pi, num_kpoints, endpoint=endpoint)
        kpoints = np.array([radius * np.cos(angles),
                            radius * np.sin(angles),
                            np.zeros(num_kpoints)], dtype=np.float64).T

        inverse_rotation = rotate_vector_to_ez(normal).T
        self.kpoints = np.einsum('nm, kn -> km', inverse_rotation, kpoints) + center


class Grid(SetOfKPoints):
    """A grid through the Brillouin zone.


    Parameters
    ----------
    num_x: int
        Number of points along the first vector that spans the grid.
    num_y: int
        Number of points along the second vector that spans the grid.
    num_z: int
        Number of points along the third vector that spans the grid.
    x_min: float
        First component of the first point of the grid. Default is -0.5.
    x_max: float
        First component of the end point of the grid. Default is 0.5.
    y_min: float
        Second component of the first point of the grid. Default is -0.5.
    y_max: float
        Second component of the end point of the grid. Default is 0.5.
    z_min: float
        Third component of the first point of the grid. Default is -0.5.
    z_max: float
        Third component of the end point of the grid. Default is 0.5.
    endpoint_x: bool
        If True, x_max is the boundary of the plane in the first direction. Default is False.
    endpoint_y: bool
        If True, y_max is the boundary of the plane in the first direction. Default is False.
    endpoint_z: bool
        If True, z_max is the boundary of the plane in the first direction. Default is False.

    Attributes
    ----------
    extent: tuple[float, float, float, float, float, float]
        This is where x_min, x_max, y_min, y_max, z_min, z_max are stored.
    kpoints: VectorList
        List of points in reciprocal space in reduced coordinates.
    num_kpoints: int
        Number of k-points.
    shape: tuple(int, int)
        This is where num_x, num_y and num_z are saved.

    Notes
    -----
    If the plane is used to compute quantities that are obtained by integrating over the Brillouin zone in
    three-dimensions, e.g. the bare susceptibility, set `endpoint_x`, `endpoint_y` and `endpoint_z` to False to avoid
    double counting the points on the boundaries of the plane and make sure that `x_min`, `x_max`, `y_min`, `y_max`,
    `z_min` and `z_max` span the full Brillouin zone.


    Examples
    --------

    We construct two grids and plot them.

    .. ipython:: python

        import matplotlib.pyplot as plt

        # Create the grids.
        num_x, num_y, num_z = 15, 15, 15
        grid1 = tp.Grid(num_x, num_y, num_z)
        grid2 = tp.Grid(num_x, num_y, num_z, x_min=0.75, x_max=1.25, y_min=0.75, y_max=1.25, z_min=0.75, z_max=1.25)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(*grid1.kpoints.T)
        ax.scatter(*grid2.kpoints.T)
        ax.set_xlabel(r'$k_x$');
        ax.set_ylabel(r'$k_y$');
        @savefig grid.png
        ax.set_zlabel(r'$k_z$');

    """

    def __init__(self,
                 num_x: int,
                 num_y: int,
                 num_z: int,
                 x_min: float = -0.5,
                 x_max: float = 0.5,
                 y_min: float = -0.5,
                 y_max: float = 0.5,
                 z_min: float = -0.5,
                 z_max: float = 0.5,
                 endpoint_x: bool = False,
                 endpoint_y: bool = False,
                 endpoint_z: bool = False) -> None:

        self.extent = (x_min, x_max, y_min, y_max, z_min, z_max)
        self.num_kpoints = num_x * num_y * num_z
        self.shape = (num_x, num_y, num_z)

        span_x = np.linspace(x_min, x_max, num_x, endpoint=endpoint_x)
        span_y = np.linspace(y_min, y_max, num_y, endpoint=endpoint_y)
        span_z = np.linspace(z_min, z_max, num_z, endpoint=endpoint_z)
        kxs, kys, kzs = np.meshgrid(span_x, span_y, span_z, indexing='ij')
        self.kpoints = np.array([kxs.flatten(), kys.flatten(), kzs.flatten()], dtype=np.float64).T


class Path(SetOfKPoints):
    """Path through reciprocal space.

    Parameters
    ----------
    nodes: VectorList
        List of (high-symmetry) points between which a path is created.
    segment_lengths: IntVector
        Number of points for each segment along the path. If None, its 100 points per segment. Default is None.

    Attributes
    ----------
    kpoints: VectorList
        List of points in reciprocal space in reduced coordinates.
    nodes: VectorList
        This is where nodes is saved.
    node_indices: IntVector
        Indices of the kpoints that correspond to the nodes.
    num_kpoints : int
        Number of kpoints.
    segment_lenghts: IntVector
        Number of kpoints between each node.

    Examples
    --------

    Let's create a path from Gamma to M to K and back to Gamma.

    .. ipython:: python

        path = tp.Path([[0, 0, 0],
                        [1 / 2, 0, 0],
                        [1 / 3, 1 / 3, 0],
                        [0, 0, 0]])

        fig, ax = plt.subplots()
        ax.plot(*path.kpoints[:, :2].T)
        ax.scatter(*path.nodes[:, :2].T)
        ax.set_xlabel(r'$k_x$')
        @savefig path.png
        ax.set_ylabel(r'$k_y$')

    """

    # TODO: put this into a static method and convert to dataclass
    def __init__(self,
                 nodes: VectorList,
                 segment_lengths: list[int] = None) -> None:

        self.kpoints = np.array([], dtype=np.float64).reshape((0, 3))
        num_nodes = len(nodes)
        self.nodes = np.array(nodes, dtype=np.float64).reshape((num_nodes, 3))

        if segment_lengths is None:
            self.segment_lengths = np.array([100] * (num_nodes - 1), dtype=np.int64)
        else:
            self.segment_lengths = np.array(segment_lengths, dtype=np.int64).reshape((num_nodes - 1,))
        self.node_indices = np.concatenate(([0], np.cumsum(self.segment_lengths)), dtype=np.int64)

        for start_point, end_point, segment_length in zip(self.nodes[:-1], self.nodes[1:], self.segment_lengths):
            kpoints_segment = np.linspace(start_point, end_point, segment_length, endpoint=False)
            self.kpoints = np.concatenate((self.kpoints, kpoints_segment), axis=0)
        self.kpoints = np.concatenate((self.kpoints, self.nodes[-1].reshape((1, 3))), axis=0)
        self.num_kpoints = len(self.kpoints)


class Plane(SetOfKPoints):
    """A plane through the Brillouin zone.


    Parameters
    ----------
    normal: Vector
        The normal of the plane.
    num_x: int
        Number of points along the first vector that spans the plane.
    num_y: int
        Number of points along the second vector that spans the plane.
    anchor: float
        Where along the normal the plane is anchored. Default is 0
    x_min: float
        First component of the first point of the plane. Default is -0.5.
    x_max: float
        First component of the end point of the plane. Default is 0.5.
    y_min: float
        Second component of the first point of the plane. Default is -0.5.
    y_max: float
        Second component of the end point of the plane. Default is 0.5.
    endpoint_x: bool
        If True, x_max is the boundary of the plane in the first direction. Default is False.
    endpoint_y: bool
        If True, y_max is the boundary of the plane in the first direction. Default is False.

    Attributes
    ----------
    anchor: float
        This is where anchor is saved.
    extent: tuple[float, float, float, float]
        This is where x_min, x_max, y_min and y_max are stored.
    kpoints: VectorList
        List of points in reciprocal space in reduced coordinates.
    normal: Vector
        This is where normal is saved.
    num_kpoints: int
        Number of k-points.
    shape: tuple(int, int)
        This is where num_x and num_y are saved.

    Notes
    -----
    If the plane is used to compute quantities that are obtained by integrating over the Brillouin zone in
    two-dimensions, e.g. the density of states of graphene, set `endpoint_x` and `endpoint_y` to False to avoid double
    counting the points on the boundaries of the plane and leave make sure `x_min`, `x_max`, `y_min` and `y_max`
    span the full Brillouin zone.


    Examples
    --------

    We construct three planes and plot them.

    .. ipython:: python

        import matplotlib.pyplot as plt

        # Create the planes.
        num_x, num_y = 31, 31
        plane_xy = tp.Plane([0, 0, 1], num_x, num_y, endpoint_x=True, endpoint_y=True)
        plane_xy_shifted = tp.Plane([0, 0, 1], num_x, num_y, anchor=0.1, endpoint_x=True, endpoint_y=True)
        plane_101 = tp.Plane([1, 0, 1], num_x, num_y, endpoint_x=True, endpoint_y=True)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for plane in [plane_xy, plane_xy_shifted, plane_101]:
            ax.scatter(*plane.kpoints.T)
        ax.set_xlabel(r'$k_x$');
        ax.set_ylabel(r'$k_y$');
        @savefig plane.png
        ax.set_zlabel(r'$k_z$');

    """

    def __init__(self,
                 normal: Vector,
                 num_x: int,
                 num_y: int,
                 anchor: float = 0.0,
                 x_min: float = -0.5,
                 x_max: float = 0.5,
                 y_min: float = -0.5,
                 y_max: float = 0.5,
                 endpoint_x: bool = False,
                 endpoint_y: bool = False) -> None:

        self.anchor = anchor
        self.extent = (x_min, x_max, y_min, y_max)
        self.normal = np.array(normal, dtype=np.float64)
        self.num_kpoints = num_x * num_y
        self.shape = (num_x, num_y)

        span_x = np.linspace(x_min, x_max, num_x, endpoint=endpoint_x)
        span_y = np.linspace(y_min, y_max, num_y, endpoint=endpoint_y)
        kxs, kys = np.meshgrid(span_x, span_y, indexing='ij')
        kpoints = np.array([kxs.flatten(), kys.flatten(), np.zeros(self.num_kpoints)], dtype=np.float64).T

        inverse_rotation = rotate_vector_to_ez(normal).T
        self.kpoints = np.einsum('nm, kn -> km', inverse_rotation, kpoints) + anchor * self.normal

