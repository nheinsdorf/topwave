from __future__ import annotations

import numpy as np
import numpy.fft as fft

from topwave.constants import K_BOLTZMANN
from topwave.fourier_coefficients import get_intracell_fourier_coefficient
from topwave.model import TightBindingModel
from topwave.set_of_kpoints import Grid
from topwave.spec import Spec
from topwave.types import ListOfRealList
from topwave.util import fermi_distribution


def get_nernst_conductivity(
        berry_curvature: ListOfRealList,
        energies: ListOfRealList,
        filling: float,
        temperature: float) -> float:
    """Computes the Nernst Conductivity.

    Parameters
    ----------
    berry_curvature: ListOfRealList
        The Berry Curvature at each k-point for all bands.
    energies: ListOfRealList
        The energies at each k-point for all bands.
    filling: float
        The energy up to which the states are considered.
    temperature: float
        The temperature.

    Returns
    -------
    float
        The Nernst conductivity at some filling.

    """

    energies * fermi_distribution(energies, temperature) + K_BOLTZMANN * temperature * np.log(1 + np.exp(energies - filling))

def get_bare_susceptibility(
        model: TightBindingModel,
        k_grid_shape: tuple[int, int, int],
        temperature: float,
        num_matsubara_frequencies: int,
        symmetrize: bool = True) -> np.ndarray[np.float64]:
    """Computes the bare susceptibility tensor for a given model.

    This uses the imaginary time representation to efficiently calculate the product of Green's functions.
    Cite Something

    Parameters
    ----------
    model: TightBindingModel
        A list of tight-binding hamiltonians on a grid that covers the Brillouin zone that is used to calculate the bare
        suscpetibility. The shape should be the shape of the grid times the dimension of the hamiltonians.
    k_grid_shape: tuple(int, int, int)
        A tuple that gives the shape of the grid in reciprocal space. It is used to reshape `hamiltonians` and perform
        the multidimensional fast Fourier transform. For two-dimensional grids chose the last number in this tuple to
        be 1.
    temperature: float
        The temperature in the units of the hopping amplitudes. 2 percent of the largest hopping amplitude is a good
        starting value.
    num_matsubara_frequencies: int
        The number of matsubara frequencies. Increase this number until convergence is reached. A good starting point
        is 256.
    symmetrize: bool
        If true, the susceptibility tensor is transformed taking into account the intracell sublattice positions to
        restore the symmetries of e.g. nonsymmorphic crystals. Default is true.

    Returns
    -------
    np.ndarray[np.float64]
        The bare susceptibility. The shape is the shape of the kpoint-grid times for indices that run over the number
        of bands.

    Notes
    -----
    Make sure that the grid that covers the BZ does not include the endpoints so that there is no double counting. And
    that the hamiltonian is reshaped properly.
    See also...

    Examples
    --------

    We calculate the bare susceptibilty of the square lattice and a single orbital.

    .. ipython:: python

        # create a two-dimensional square lattice
        from pymatgen.core.structure import Structure
        a, vacuum = 1, 10
        lattice = [[a, 0, 0], [0, a, 0], [0, 0, vacuum]]
        struc = Structure.from_spacegroup(1, lattice, ['Cu'], [np.zeros(3)])

        # Construct a Model instance and set nearest neighbor couplings
        model = tp.TightBindingModel(struc)
        model.generate_couplings(1, 1)
        t1 = -1
        model.set_coupling(a, t1, "distance")
        model.show_couplings()

    """

    grid = Grid(num_x=k_grid_shape[0], num_y=k_grid_shape[1], num_z=k_grid_shape[2],
                x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1,
                endpoint_x=False, endpoint_y=False, endpoint_z=False)
    hamiltonians = Spec.get_tightbinding_hamiltonian(model, grid)
    num_bands = hamiltonians.shape[-1]
    hamiltonians = hamiltonians.reshape((*k_grid_shape, num_bands, num_bands))

    positive_frequencies = np.pi * np.arange(1, 2 * num_matsubara_frequencies + 1, 2)
    matsubara_frequencies = 1j * temperature * np.concatenate((positive_frequencies, -positive_frequencies[::-1]),
                                                              dtype=np.complex128)

    Greens_functions = np.linalg.inv(np.einsum('f, nm -> fnm',
                                               matsubara_frequencies,
                                               np.eye(num_bands))[:, np.newaxis, np.newaxis, np.newaxis, :, :] - hamiltonians)
    Greens_functions = fft.fft(Greens_functions, axis=0)

    temperature_dependence = temperature * np.exp(-1j * np.pi * np.linspace(0, 1, 2 * num_matsubara_frequencies,
                                                                            endpoint=False))
    Greens_functions = np.einsum('f, fhklnm -> fhklnm', temperature_dependence,  Greens_functions)

    Greens_functions = fft.ifftn(Greens_functions, axes=[1, 2, 3])

    Greens_functions_negative_times = np.roll(np.roll(np.roll(np.roll(-1 * Greens_functions[::-1, ::-1, ::-1, ::-1],
                                                                      1, axis=0),
                                                              1, axis=1),
                                                      1, axis=2),
                                              1, axis=3)
    Greens_functions_negative_times[0] *= -1
    product_Greens_functions = -np.einsum('fhklmn, fhklop -> fhklmnop',
                                          Greens_functions,
                                          Greens_functions_negative_times) / temperature
    susceptibility = (fft.fftn(fft.ifft(product_Greens_functions, axis=0)
                               , axes=[1, 2, 3]))
    if not symmetrize:
        return  susceptibility

    intracell_phase_factors = np.array([get_intracell_fourier_coefficient(site, grid.kpoints)
                                        for site in model.structure], dtype=np.complex128).T
    intracell_phase_factors = intracell_phase_factors.reshape((*grid.shape, num_bands))[..., np.newaxis] \
                              * np.eye(len(model.structure))
    symmetric_susceptibility = np.einsum('whklabcd, hklbc, hklad -> whklabcd',
                                         fft.fftshift(susceptibility, axes=[1, 2, 3]),
                                         intracell_phase_factors,
                                         np.conj(intracell_phase_factors))
    return symmetric_susceptibility


