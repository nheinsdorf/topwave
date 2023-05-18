from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import det

from topwave.types import IntVector, Matrix

if TYPE_CHECKING:
    from topwave.spec import Spec

__all__ = ["get_berry_phase", "get_bosonic_wilson_loop", "get_fermionic_wilson_loop"]

def get_berry_phase(loop_operator: Matrix) -> float:
    """Computes the Berry phase of a Wilson loop:

    .. math:: \phi = - \sum \operatorname{Im} \ln \det M^{(\Lambda_{i}, \Lambda_{i+1})},


    with

    .. math:: M^{(\Lambda_{i}, \Lambda_{i+1})}_{mn} = \langle u_m^{(\Lambda_{i})} u_n^{(\Lambda_{i+1})} \rangle

    Examples
    --------
    Compute the Berry phase of something.


    """

    return -1 * np.angle(det(loop_operator))


def get_bosonic_wilson_loop(spectrum: Spec, band_indices: IntVector) -> Matrix:
    """Constructs the Wilson loop operator of a bosonic spectrum.

    """

    pass

def get_fermionic_wilson_loop(spectrum: Spec, band_indices: IntVector) -> Matrix:
    """Constructs the Wilson loop operator of a fermionic spectrum.

    For a spectrum at Nk k-points, the inner product of Nk - 1 eigenfunctions for a given selection of
    states is evaluated. The ordering is the same as that of the k-points.

    .. admonition:: Careful!
        :class: warning

        Typically, **closed** Wilson loops of the **occupied** states are the desired quantities. Make sure the
        last and first k-point of the spectrum are the same, and all selected bands are
        separated in energy (**nondegenerate**).

    Parameters
    ----------
    spectrum: Spec
        The spectrum that contains the eigenfunctions of the model.
    band_indices: IntVector
        List of band indices that are selected to compute the Wilson loop operator.

    """

    psi = spectrum.psi[:, :, band_indices]

    # check whether start and end k-point are the same and impose closed loop
    if np.all(np.isclose(spectrum.kpoints[0], spectrum.kpoints[-1])):
        psi[0] = psi[-1]
    else:
        # implement the case where they are connected by a reciprocal vector
        # https://github.com/bellomia/PythTB/blob/master/pythtb.py
        # see 'impose_pbc'-method
        pass

    # construct bra-eigenvectors for k+1
    psi_left = np.roll(np.conj(psi), 1, axis=0)

    # compute num_k - 1 overlaps
    loop = np.einsum('knm, knl -> kml', psi_left[1:], psi[1:])

    # do the SVD cleanup?
    # take the product of the matrices to compute the wilson loop operator
    return np.linalg.multi_dot(loop)

