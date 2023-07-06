from __future__ import annotations


import numpy as np
from pymatgen.core.sites import PeriodicSite

from topwave.coupling import Coupling
from topwave.types import ComplexList, VectorList


#TODO: implement the other convention of FT that takes into account the intra cell positions of sites.
def get_intracell_fourier_coefficient(site: PeriodicSite,
                                      kpoint: VectorList) -> ComplexList:
    """Returns the phase factor from different sublattice positions.



    Parameters
    ----------
    site: PeriodicSite
        The `pymatgen.core.sites.PeriodicSite` for which the coefficient is computed.
    kpoint: VectorList
        The kpoints at which the coefficient is computed.

    Returns
    -------
    ComplexList
        The intracell Fourier coefficient at the given k-point of the site.

    """

    return np.exp(-2j * np.pi * np.einsum('x, kx -> k', site.frac_coords, kpoint))

def get_periodic_fourier_coefficient(coupling: Coupling,
                                     kpoint: VectorList) -> ComplexList:
    """Returns the Fourier coefficient of a coupling to get a Brillouin zone periodic Hamiltonian.


    This is the convention of Fourier transform where the positions of the atomic sites within one unit cell
    are not taken into account. This should be used for Wilson loop calculations. In this convention, the physical
    representations of point group symmetries have no k-dependence. (CHECK!)

    Parameters
    ----------
    coupling: Coupling
        The `topwave.coupling.Coupling` for which the coefficient is computed.
    kpoint: VectorList
        The kpoints at which the coefficient is computed.

    Returns
    -------
    ComplexList
        The Fourier coefficient at the given k-point of the coupling.

    """

    return np.exp(-2j * np.pi * np.einsum('x, kx -> k', coupling.lattice_vector, kpoint))


def get_periodic_fourier_derivative(coupling: Coupling,
                                    kpoint: VectorList,
                                    direction: str) -> ComplexList:
    """Returns the derivative of the Fourier coefficient w.r.t. a list of crystal momenta.


    This is used to compute the element-wise derivatives of the Hamiltonian w.r.t. crystal momentum. These
    tangent matrices are useful for computing Berry curvatures or transport properties like the Nernst effect.

    Parameters
    ----------
    coupling: Coupling
        The `topwave.coupling.Coupling` for which the derivative is computed.
    kpoint: VectorList
        The kpoints at which the derivative is computed.
    direction: str
        Which crystal momentum is used for the derivative. Options are 'x', 'y' and 'z'.

    Returns
    -------
    ComplexList
        The derivative at the given k-point of the coupling.

    """

    index = 'xyz'.find(direction)
    coefficient = get_periodic_fourier_coefficient(coupling, kpoint)
    inner_derivative = -2j * np.pi * coupling.lattice_vector[index]
    return inner_derivative * coefficient




