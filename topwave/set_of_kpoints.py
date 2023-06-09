from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from pymatgen.core.structure import Structure

from topwave.types import Vector, VectorList
from topwave.util import rotate_vector_to_ez

__all__ = ["Circle", "SetOfKPoints", "Path"]

class SetOfKPoints(ABC):
    """Base class that is used to parameterize paths and manifolds in reciprocal space.

    This is an **abstract** base class. Use its child classes to instantiate a model.

    Attributes
    ----------
    kpoints : VectorList
        List of points in reciprocal space in reduced coordinates.
    num_kpoints : int
        Number of kpoints.

    """

    def __init__(self):
        self.kpoints = self.get_kpoints()
        self.num_kpoints = len(self.kpoints)

    @abstractmethod
    def _get_kpoints(self) -> VectorList:
        return self.kpoints

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

    def _get_kpoints(self) -> VectorList:
        return self.kpoints


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
    def __init__(self, nodes: VectorList, segment_lengths: list[int] = None) -> None:

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

    def _get_kpoints(self) -> VectorList:
        return self.kpoints


class Plane(SetOfKPoints):
    """A Plane in reciprocal space.

    Parameters
    ----------
    x_min: float
        Radius in reduced reciprocal lattice units of the circle.
    center: Vector
        Where in reciprocal space the plane is centered.
    normal: Vector
        The normal of the plane the circle lies in.
    num_kpoints_x: int
        The number of k-points along the first direction of the plane. Default is 100.
    num_kpoints_y: int
        The number of k-points along the second direction of the plane. Default is 100.
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

    def _get_kpoints(self) -> VectorList:
        return self.kpoints




