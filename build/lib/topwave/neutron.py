import numpy as np
from numpy.linalg import norm

from topwave.util import bose_distribution, gaussian
# CHECK for model.dim
class NeutronScattering:
    """Base class to calculate observables that are measured in neutron scattering experiments.

    Parameters
    ----------
    spec : topwave.spec.Spec
        Spinwave spectrum.
    energy_bins : list or numpy.ndarray
        List of energy_bins for which the intensity map is computed. If None, 501 energy_bins from 0 to 1.1 * max(spec)
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
    energy_bins : numpy.ndarray
        This is where energy_bins is stored.
    intensity_map : numpy.ndarray
        This is where the intensity map of the cross section is stored. The shape is (NK, NE), where NE is the
        number of provided energy_bins.
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

    def __init__(self, spec, energy_bins=None, resolution=0.05, temperature=0):
        self.spec = spec
        self.component = 'S_perp'
        self.SS = self.get_spin_spin_expectation_val()
        self.cross_section = self.get_cross_section()
        self.intensity_map, self.energy_bins = self.get_intensity_map(energy_bins, resolution, temperature)
        self.resolution = resolution
        self.temperature = temperature

    def __getattr__(self, item):
        try:
            return getattr(self.spec, item)
        except:
            raise AttributeError(f"NeutronScattering object has no attribute {item}.")


    def get_cross_section(self):
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
        hkl = (self.k_points_xyz.T / norm(self.k_points_xyz, axis=1)).T
        q_perp = np.eye(3) - np.einsum('ka, kb -> kab', hkl, hkl)

        # Calculate the perpendicular component of the (symmetric part of the) dynamical structure factor.
        return np.real(np.einsum('kab, knab -> kn', q_perp, 0.5 * (self.SS + self.SS.swapaxes(2, 3))))

    def get_intensity_map(self, energy_bins=None, resolution=0.05, temperature=0):
        """Converts a correlation function to an intensity map on a provided set of energy bins.

        Parameters
        ----------
        energy_bins : list or numpy.ndarray
            List of energy_bins for which the intensity map is computed. If None, 501 energy_bins from 0 to 1.1 * max(spec)
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
        The convolved cross_section at the provided energy_bins and the energy_bins.
        """

        dim = len(self.model.structure)
        if energy_bins is None:
            energy_bins = np.linspace(0, 1.1 * np.max(self.energies), 501)

        # NOTE: this should be (maybe) complex for other components of the dynamical structure factor
        intensity_map = np.zeros((len(self.k_points), len(energy_bins)), dtype=float)
        for k, (E_k, S_k) in enumerate(zip(self.energies[:, :dim], self.cross_section[:, :dim])):
            bose_factors = bose_distribution(E_k, temperature=temperature) + 1
            for (mean, intensity, bose_factor) in zip(E_k, S_k, bose_factors):
                intensity_map[k, :] += intensity * gaussian(energy_bins, mean, resolution) * bose_factor

        return intensity_map, energy_bins

    def get_spin_spin_expectation_val(self):
        """
        Calculates the local spin-spin expectation values for a given set of k_points.
        """

        model = self.model
        k_points = self.k_points
        psi_k = self.psi
        dim = len(self.model.structure)

        struc = self.model.structure

        # calculate the phase factors for all sites
        phases = np.zeros((len(k_points), dim), dtype=complex)
        us = np.zeros((dim, 3), dtype=complex)
        for _, site in enumerate(struc.sites):
            mu = np.sqrt(norm(site.properties['magmom'] / 2))
            phases[:, _] = mu * np.exp(-1j * np.einsum('ki, i -> k', k_points, site.frac_coords) * 2 * np.pi)
            us[_, :] = site.properties['Rot'][:, 0] + 1j * site.properties['Rot'][:, 1]

        phasesL = np.transpose(np.tile(np.concatenate((phases, phases), axis=1), (3, 3, 2 * dim, 1, 1)), (3, 2, 4, 0, 1))
        phasesR = np.conj(phasesL.swapaxes(1, 2))

        usL = np.transpose(np.tile(np.concatenate((us, np.conj(us)), axis=0), (2 * dim, 3, 1, 1)), (0, 2, 3, 1))
        usL = np.tile(usL, (len(k_points), 1, 1, 1, 1))
        usR = np.conj(np.transpose(usL, (0, 2, 1, 4, 3)))

        psiR = np.transpose(np.tile(psi_k, (3, 3, 1, 1, 1)), (2, 3, 4, 0, 1))
        psiL = np.conj(psiR.swapaxes(1, 2))

        S_k = np.sum(usL * phasesL * psiL, axis=2) * np.sum(usR * phasesR * psiR, axis=1)

        return S_k