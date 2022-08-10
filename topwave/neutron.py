import numpy as np
from numpy.linalg import norm

from topwave.util import bose_distribution, gaussian

class NeutronScattering:
    """Base class to calculate observables that are measured in neutron scattering experiments.

    Parameters
    ----------
    spec : topwave.spec.Spec
        Spinwave spectrum.
    energies : list or numpy.ndarray
        List of energies for which the intensity map is computed. If None, 501 energies from 0 to 1.1 * max(spec)
        are used. Default is None.
    resolution : float
        Standard deviation of the normal distribution that is convolved with the intensity map
        to account for finite instrument resolution.
    temperature : float
        Temperature given in Kelvin that is used to calculate the Bose-Einstein distribution at the provided energy bins.

    Attributes
    ----------
    spec : topwave.spec.Spec
        This is where spec is stored.
    cross_section : numpy.ndarray
        This is where the computed cross section of the spectrum is stored. The shape is (NK, N).
    energies : numpy.ndarray
        This is where energies is stored.
    intensity_map : numpy.ndarray
        This is where the intensity map of the cross section is stored. The shape is (NK, NE), where NE is the
        number of provided energies.
    resolution : float
        This is where resolution is stored.
    temperature : float
        This is where temperature is stored.

    Methods
    -------
    get_cross_section(spec)
        Computes the neutron scattering cross section of unpolarized neutrons.
    get_intensity_map(bin_edges, temperature, resolution)
        Creates an intensity map using the cross section and a list of energy bins.

    """

    def __init__(self, spec, energies=None, resolution=0.05, temperature=0):
        self.spec = spec
        self.component = 'S_perp'
        self.cross_section = self.get_cross_section(spec)
        self.intensity_map, self.energies = self.get_intensity_map(energies, resolution, temperature)
        self.resolution = resolution
        self.temperature = temperature

        # save everything in spec's 'neutron'-dict
        spec.neutron['component'] = self.component
        spec.neutron['cross_section'] = self.cross_section
        spec.neutron['energies'] = self.energies
        spec.neutron['intensity_map'] = self.intensity_map
        spec.neutron['resolution'] = self.resolution
        spec.neutron['temperature'] = self.temperature

    def __getattr__(self, item):
        try:
            return getattr(self.spec, item)
        except:
            raise AttributeError(f"NeutronScattering object has no attribute {item}.")

    @staticmethod
    def get_cross_section(spec):
        """Calculates the neutron scattering cross section of unpolarized neutrons.

        The neutron scattering cross section is defined as the perpendicular part of the symmetrized
        dynamical structure factor.

        Parameters
        ----------
        spec : topwave.spec.Spec
            The spin-wave spectrum that contains the spin-spin correlation function.

        Returns
        -------
        The perpendicular part of the (symmetrized) dynamical structure factor.

        """

        # Normalize all the scattering vectors.
        hkl = (spec.KS_xyz.T / norm(spec.KS_xyz, axis=1)).T
        q_perp = np.eye(3) - np.einsum('ka, kb -> kab', hkl, hkl)

        # Calculate the perpendicular component of the (symmetric part of the) dynamical structure factor.
        return np.real(np.einsum('kab, knab -> kn', q_perp, 0.5 * (spec.SS + spec.SS.swapaxes(2, 3))))

    def get_intensity_map(self, energies=None, resolution=0.05, temperature=0):
        """Converts a correlation function to an intensity map on a provided set of energy bins.

        Parameters
        ----------
        energies : list or numpy.ndarray
            List of energies for which the intensity map is computed. If None, 501 energies from 0 to 1.1 * max(spec)
            are used. Default is None.
        resolution : float
            Standard deviation of the normal distribution that is convolved with the intensity map
            to account for finite instrument resolution. Because the intensity map is just a sum of delta-peaks,
            the convolution is trivial and is just the sum of normal distributions centered at the peak positions.
            Default is 50 meV.
        temperature : float
            Temperature given in Kelvin. It is used to construct the Bose factor. If 0, the Bose factor is not
            taken into account. Default is 0.

        Returns
        -------
        The convolved cross_section at the provided energies and the energies.
        """

        if energies is None:
            energies = np.linspace(0, 1.1 * np.max(self.E), 501)

        # NOTE: this should be (maybe) complex for other components of the dynamical structure factor
        intensity_map = np.zeros((self.NK, len(energies)), dtype=float)
        for k, (E_k, S_k) in enumerate(zip(self.E[:, :self.N], self.cross_section[:, :self.N])):
            bose_factors = bose_distribution(E_k, temperature=temperature) + 1
            for (mean, intensity, bose_factor) in zip(E_k, S_k, bose_factors):
                intensity_map[k, :] += intensity * gaussian(energies, mean, resolution) * bose_factor

        return intensity_map, energies
