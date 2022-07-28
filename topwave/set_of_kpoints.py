import numpy as np

class SetOfKPoints:
    """Parent class for all SetOfKPoint objects

    Parameters
    ----------
    kpoints : numpy.ndarray
        Array of shape (num_kpoints x 3) specifying num_kpoints k-points in fractional, reciprocal coordinates.
    shape : tuple
        The shape of the kpoints in each direction, e.g. a grid or a plane. Default is None.
    Attributes
    ----------
    kpoints : numpy.ndarray
        This is where kpoints is stored.
    num_kpoints : int
        Number of kpoints.
    shape : tuple
        This is where shape is stored. If None, it is set to (num_kpoints, 3)
    """
    def __init__(self, kpoints, shape=None):
        try:
            self.kpoints = np.array(kpoints, dtype=float).reshape((len(kpoints), 3))
        except:
            raise TypeError('SetOfKPoints needs to be a list (or array) of three-dimensional kpoints.')
        if shape is not None:
            try:
                self.kpoints.reshape((*shape, 3))
            except:
                raise ValueError('Shape does not match the number of provided kpoints.')

class Grid(SetOfKPoints):
    """Grid of kpoints that span the whole three-dimensional Brillouin Zone.

    Parameters
    ----------
    shape : tuple
        Tuple of three integers that specify the number of kpoints in each direction of the reciprocal lattice.
    struc : pymatgen.core.Structure
        Pymatgen Structure that is used to construct the irreducible grid of kpoints.
    space_group : int
        International space group number. If None, the space group symmetry is determined from struc.
        Default is None.
    """
