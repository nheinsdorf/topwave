import numpy as np
from numpy.linalg import norm

class NeutronScattering:
    """Base class to calculate observables that are measured in neutron scattering experiments.

    Parameters
    ----------
    spec : topwave.spec.Spec
        Spinwave spectrum.

    Attributes
    ----------
    spec : topwave.spec.Spec
        This is where spec is stored.

    Methods
    -------
    get_cross_section(spec)
        Computes the neutron scattering cross section of unpolarized neutrons.

    """

    def __init__(self, spec):
        self.spec = spec
        self.get_cross_section(self.spec)

    def __getattr__(self, item):
        try:
            return getattr(self.spec, item)
        except:
            raise AttributeError(f"NeutronScattering object has no attribute {item}.")

    def get_cross_section(self, spec):
        """Calculates the neutron scattering cross section of unpolarized neutrons.

        The neutron scattering cross section is defined as the perpendicular part of the symmetrized
        dynamical structure factor.

        Parameters
        ----------
        spec : topwave.spec.Spec
            The spin-wave spectrum that contains the spin-spin correlation function. The output will be stored in
            the 'S_perp'-attribute of spec.

        """

        # Normalize all the scattering vectors.
        hkl = (spec.KS_xyz.T / norm(spec.KS_xyz, axis=1)).T
        q_perp = np.eye(3) - np.einsum('ka, kb -> kab', hkl, hkl)

        # Calculate the perpendicular component of the (symmetric part of the) dynamical structure factor.
        spec.S_perp = np.real(np.einsum('kab, knab -> kn', q_perp, 0.5 * (spec.SS + spec.SS.swapaxes(2, 3))))

    def get_intensity_map(self, energies):
        """Converts a correlation function to an intensity map on a provided set of energy bins.

        Parameters
        ----------
        energies : list
            List of monotonically increasing energy values that will be used as bins.

         """






    