from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from pymatgen.core.structure import Structure

from topwave.types import IntVector, VectorList

__all__ = ["SetOfKPoints", "Path"]

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


class Path(SetOfKPoints):
    """Path through reciprocal space.

    Parameters
    ----------
    nodes: VectorList
        List of (high-symmetry) points between which a path is created.
    segment_lengths: IntVector
        Number of points for each segment along the path. If None, its 100 points per segment. Default is None.

    Examples
    --------

    Let's create a path

    .. ipython:: python

        print('lol')

    """

    # TODO: put this into a static method and convert to dataclass
    def __init__(self, nodes: VectorList, segment_lengths: list[int] = None):

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






