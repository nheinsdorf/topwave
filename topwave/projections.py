import numpy as np

from topwave.spec import Spec
from topwave.types import ListOfRealList


# NOTE: implement orbital
def get_projections(spec: Spec,
                    projections: dict) -> ListOfRealList:
    """Returns the projector on a given set of attribute value pairs.

    Parameters
    ----------
    spec: Spec
        The spectrum that is projected.
    projections: dict
        A dictionary that contains all the projections as key-value pairs. Possible keys are
        'orbital', 'spin', 'sublattice', 'unit_cell_x', 'unit_cell_y' and 'unit_cell_z'.

    Returns
    -------
    ListOfRealList
        The projections.

    """

    dimension = len(spec.psi[0])

    is_spinless =  not spec.model.check_if_spinful() or spec.model._is_spin_polarized
    num_spins = 1 if is_spinless else 2

    num_sublattices = int(dimension / num_spins / np.product(spec.model.scaling_factors))

    wavefunctions = spec.psi.reshape((spec.kpoints.num_kpoints,
                                      *spec.model.scaling_factors,
                                      num_sublattices,
                                      num_spins,
                                      dimension))

    probabilities = np.real(np.conj(wavefunctions) * wavefunctions)

    unit_cell_x = projections.get('unit_cell_x', np.arange(spec.model.scaling_factors[0]))
    unit_cell_y = projections.get('unit_cell_y', np.arange(spec.model.scaling_factors[1]))
    unit_cell_z = projections.get('unit_cell_z', np.arange(spec.model.scaling_factors[2]))
    sublattice = projections.get('sublattice', np.arange(num_sublattices))
    spin = projections.get('spin', np.arange(num_spins))

    unit_cell_projection = probabilities[:, unit_cell_x][:, :, unit_cell_y][:, :, :, unit_cell_z]
    sublattice_projection = unit_cell_projection[:, :, :, :, sublattice]
    spin_projection = sublattice_projection[:, :, :, :, :, spin]

    return spin_projection.sum(axis=1).sum(axis=1).sum(axis=1).sum(axis=1).sum(axis=1)