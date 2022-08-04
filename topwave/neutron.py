import numpy as np
from numpy.linalg import norm

class NeutronScattering:
    """Base class to calculate observables that are measured in neutron scattering experiments.

    Parameters
    ----------
    spec : topwave.spec.Spec
        Spinwave spectrum.
    energy_bins : list
        List of floats that define energy bins that are used to create an intensity map using the cross section.
        If None, 501 bins from 0 to 1.1 * max(spec) are used. Default is None.

    Attributes
    ----------
    spec : topwave.spec.Spec
        This is where spec is stored.

    Methods
    -------
    get_cross_section(spec)
        Computes the neutron scattering cross section of unpolarized neutrons.
    get_intensity_map(energy_bins)
        Creates an intensity map using the cross section and a list of energy bins.

    """

    def __init__(self, spec, energy_bins=None):
        self.spec = spec
        self.cross_section = self.get_cross_section(spec)
        self.intensity_map, self.energy_bins = self.get_intensity_map(energy_bins)

        # save everything in spec's 'neutron'-dict
        spec.neutron['component'] = 'S_perp'
        spec.neutron['cross_section'] = self.cross_section
        spec.neutron['energy_bins'] = self.energy_bins
        spec.neutron['intensity_map'] = self.intensity_map

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

    def get_intensity_map(self, energy_bins=None):
        """Converts a correlation function to an intensity map on a provided set of energy bins.

        Parameters
        ----------
        energy_bins : list
            List of monotonically increasing floats that define energy bins that are used to create an intensity map
            using the cross section. If None, 501 bins from 0 to 1.1 * max(spec) are used. Default is None.

        Returns
        The cumulated cross_section w.r.t. to the provided bins at each points and the energy bins.
         """

        if energy_bins is None:
            energy_bins = np.linspace(0, 1.1 * np.max(self.E), 501)
        self.energy_bins = energy_bins

        # NOTE: this should be complex for other components of the dynamical structure factor
        intensity_map = np.zeros((self.NK, len(energy_bins) - 1), dtype=float)
        for _, (E_k, S_k) in enumerate(zip(self.E[:, :self.N], self.cross_section[:, :self.N])):
            hist, bin_edges = np.histogram(E_k, energy_bins, weights=S_k)
            intensity_map[_, :] = hist

        return intensity_map, bin_edges







    