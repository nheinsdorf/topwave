from __future__ import annotations

import numpy as np
import numpy.fft as fft
from topwave.types import SquareMatrixList


def get_bare_susceptibility(
        hamiltonians: SquareMatrixList,
        k_grid_shape: tuple[int, int, int],
        temperature: float,
        num_matsubara_frequencies: int) -> np.ndarray[np.float64]:
    """Computes the bare susceptibility tensor for a given spectrum.

    This uses the imaginary time representation to efficiently calculate the product of Green's functions.
    Cite Something

    Parameters
    ----------
    hamiltonians: SquareMatrixList
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

    hamiltonians = np.array(hamiltonians, dtype=np.complex128)
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
    product_Greens_functions = -np.einsum('fhklmn, fhklop -> fhklmnop', Greens_functions, Greens_functions_negative_times)
    susceptibility = (fft.fftn(fft.ifft(product_Greens_functions, axis=0)
                               , axes=[1, 2, 3]))

    return susceptibility


