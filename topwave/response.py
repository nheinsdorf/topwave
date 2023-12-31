from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import numpy.fft as fft

from topwave.constants import K_BOLTZMANN
from topwave.fourier_coefficients import get_intracell_fourier_coefficient
from topwave.model import TightBindingModel
from topwave.set_of_kpoints import Grid
from topwave.spec import Spec
from topwave.types import ListOfRealList, SquareMatrix
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

@dataclass
class Susceptibility:
    """Class for calculations of charge and spin susceptibilities and doing random phase approximation.

    Examples
    --------

    Do random phase approximation on a square lattice.

    .. ipython:: python

        print('Coming soon!')

    """

    model: TightBindingModel
    num_matsubara_frequencies: int
    k_grid_shape: tuple[int, int, int]
    temperature: float
    bare_susceptibility: np.ndarray[np.float64] = field(init=False)
    grid: Grid = field(init=False)

    def __post_init__(self) -> None:
        self.grid = Grid(num_x=self.k_grid_shape[0], num_y=self.k_grid_shape[1], num_z=self.k_grid_shape[2],
                         x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1,
                         endpoint_x=False, endpoint_y=False, endpoint_z=False)
        self.grid = Grid(num_x=self.k_grid_shape[0], num_y=self.k_grid_shape[1], num_z=self.k_grid_shape[2],
                         x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=-0.5, z_max=0.5,
                         endpoint_x=False, endpoint_y=False, endpoint_z=False)
        self.bare_susceptibility = self.get_bare_susceptibility(self.model,
                                                                self.grid,
                                                                self.num_matsubara_frequencies,
                                                                self.temperature)

    def contract(self,
                 operator_left: SquareMatrix = None,
                 operator_right: SquareMatrix = None,
                 intracell_phase_factors: bool = True) -> np.ndarray[np.float64]:
        """Contracts the susceptibility rank four tensor with operators.

        Parameters
        ----------
        operator_left: SquareMatrix
            The left operator. If None, the identity matrix is used. Default is None.
        operator_right: SquareMatrix
            The right operator. If None, the identity matrix is used. Default is None.
        intracell_phase_factors: bool
            If True, the intracell phase factors are accounted for. Default is True.

        Returns
        -------
        np.ndarray[np.float64]
            The bare susceptibility contracted to a rank two tensor (at each frequency and k-points).

        """

        num_bands = self.bare_susceptibility.shape[-1]
        operator_left = np.eye(num_bands) if operator_left is None else operator_left
        operator_right = np.eye(num_bands) if operator_right is None else operator_right

        contracted_susceptibility = np.einsum('ab, whklabcd, cd -> whklad',
                                              operator_left,
                                              self.bare_susceptibility.transpose((0, 1, 2, 3, 4, 6, 5, 7)),
                                              operator_right)
        if not intracell_phase_factors:
            return contracted_susceptibility

        intracell_phase_factors = np.array([get_intracell_fourier_coefficient(site, self.grid.kpoints)
                                            for site in self.model.structure], dtype=np.complex128).T.reshape((*self.grid.shape, num_bands))
        # intracell_phase_factors = fft.fftshift(intracell_phase_factors, axes=[0, 1, 2])

        return np.einsum('hkla, whklab, hklb -> whklab',
                         intracell_phase_factors,
                         contracted_susceptibility,
                         np.conj(intracell_phase_factors))

    @staticmethod
    def get_bare_susceptibility(
            model: TightBindingModel,
            grid: Grid,
            num_matsubara_frequencies: int,
            temperature: float) -> np.ndarray[np.float64]:
        """Computes the bare susceptibility tensor for a given model.

        This uses the imaginary time representation to efficiently calculate the product of Green's functions.
        Cite Something

        Parameters
        ----------
        model: TightBindingModel
            A list of tight-binding hamiltonians on a grid that covers the Brillouin zone that is used to calculate the bare
            suscpetibility. The shape should be the shape of the grid times the dimension of the hamiltonians.
        grid: Grid
            A grid of k-points on which the susceptibility is computed.
        num_matsubara_frequencies: int
            The number of matsubara frequencies. Increase this number until convergence is reached. A good starting point
            is 256.
        temperature: float
            The temperature in the units of the hopping amplitudes. 2 percent of the largest hopping amplitude is a good
            starting value.

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


        hamiltonians = Spec.get_tightbinding_hamiltonian(model, grid)
        num_bands = hamiltonians.shape[-1]
        hamiltonians = hamiltonians.reshape((*grid.shape, num_bands, num_bands))

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
        susceptibility = fft.fftn(fft.ifft(product_Greens_functions, axis=0), axes=[1, 2, 3])
        # if not symmetrize:
        return susceptibility

        # intracell_phase_factors = np.array([get_intracell_fourier_coefficient(site, grid.kpoints)
        #                                     for site in model.structure], dtype=np.complex128).T
        # intracell_phase_factors = intracell_phase_factors.reshape((*grid.shape, num_bands))[..., np.newaxis] \
        #                           * np.eye(len(model.structure))
        # symmetric_susceptibility = np.einsum('whklabcd, hklbc, hklad -> whklabcd',
        #                                      fft.fftshift(susceptibility, axes=[1, 2, 3]),
        #                                      intracell_phase_factors,
        #                                      np.conj(intracell_phase_factors))
        # symmetric_susceptibility = np.einsum('hklbc, whklabcd, hklad -> whklabcd',
        #                                      intracell_phase_factors,
        #                                      fft.fftshift(susceptibility, axes=[1, 2, 3]),
        #                                      np.conj(intracell_phase_factors))
        # return symmetric_susceptibility


