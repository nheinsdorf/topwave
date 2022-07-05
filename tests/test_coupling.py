import unittest

import numpy as np
from numpy.linalg import norm
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Lattice
from scipy.spatial.transform import Rotation

from topwave.coupling import Coupling
from topwave.util import rotate_vector_to_ez

class ModelTest(unittest.TestCase):
    def setUp(self):
        """ Instance of an arbitrary coupling between two PeriodicSites in a cubic lattice."""
        a = 8.908
        lattice = Lattice.from_parameters(a=a, b=a, c=a, alpha=90, beta=90, gamma=90)
        site1 = PeriodicSite('Cu', [0, 0, 0], lattice)
        site1.properties['id'] = 0
        site2 = PeriodicSite('Cu', [0.5, 0.5, 0], lattice)
        site2.properties['id'] = 1
        self.params = {'site1': site1,
                       'site2': site2,
                       'cplid': 0,
                       'symid': 0,
                       'symop': SymmOp(np.eye(4)),
                       'R': np.array([2., -1., 0.])}

    def set_moments_and_couplings(self, setup_params):
        """ This is a convenience function for testing the 'get_sw_matrix_elements' function."""

        moment1 = setup_params['moment1']
        moment2 = setup_params['moment2']
        self.params['site1'].properties['magmom'] = moment1
        self.params['site2'].properties['magmom'] = moment2
        self.params['site1'].properties['Rot'] = rotate_vector_to_ez(moment1)
        self.params['site2'].properties['Rot'] = rotate_vector_to_ez(moment2)
        coupling = Coupling(**self.params)
        coupling.strength = setup_params['strength']
        coupling.DM = setup_params['DM']
        return coupling

    def test_init_case1(self):
        """ Check that u and v vectors are None if the sites do not have the 'Rot' key in properties."""
        with self.assertRaises(KeyError):
            _ = self.params['site1'].properties['Rot']
        with self.assertRaises(KeyError):
            _ = self.params['site2'].properties['Rot']
        coupling = Coupling(**self.params)
        self.assertIsNone(coupling.u1)
        self.assertIsNone(coupling.u2)
        self.assertIsNone(coupling.v1)
        self.assertIsNone(coupling.v2)

    def test_init_case2(self):
        """ Check that the distance between the two sites is calculated correctly."""
        coupling = Coupling(**self.params)
        delta_frac = self.params['site2'].frac_coords - self.params['site1'].frac_coords + self.params['R']
        distance = norm(self.params['site1'].lattice.matrix.T @ delta_frac)
        self.assertAlmostEqual(coupling.D, distance)

    def test_init_case3(self):
        """ Check that coupling is instantiated with strength = 0 and DM = [0, 0, 0]."""
        coupling = Coupling(**self.params)
        self.assertEqual(coupling.strength, 0)
        np.testing.assert_equal(coupling.DM, [0, 0, 0])

    def test_init_case4(self):
        """ Check that the instance attributes match with the data from the dataframe."""
        coupling = Coupling(**self.params)
        self.assertEqual(coupling.DF['symid'][0], coupling.SYMID)
        self.assertEqual(coupling.DF['symop'][0], coupling.SYMOP.as_xyz_string())
        np.testing.assert_equal(coupling.DF['delta'][0], coupling.DELTA)
        np.testing.assert_equal(coupling.DF['R'][0], coupling.R)
        self.assertEqual(coupling.DF['dist'][0], coupling.D)
        self.assertEqual(coupling.DF['i'][0], coupling.I)
        self.assertEqual(coupling.DF['at1'][0], str(coupling.SITE1.species))
        self.assertEqual(coupling.DF['j'][0], coupling.J)
        self.assertEqual(coupling.DF['at2'][0], str(coupling.SITE2.species))
        self.assertEqual(coupling.DF['strength'][0], coupling.strength)
        np.testing.assert_equal(coupling.DF['DM'][0], coupling.DM)

    def test_get_uv_case1(self):
        """ Check that the u and v vectors are calculated correctly."""
        rot1 = np.eye(3)
        rot2 = Rotation.from_rotvec([0, 0, np.radians(90)]).as_matrix()
        self.params['site1'].properties['Rot'] = rot1
        self.params['site2'].properties['Rot'] = rot2
        u1, v1, u2, v2 = rot1[:, 0] + 1j * rot1[:, 1], rot1[:, 2], rot2[:, 0] + 1j * rot2[:, 1], rot2[:, 2]
        coupling = Coupling(**self.params)
        np.testing.assert_almost_equal(coupling.u1, u1)
        np.testing.assert_almost_equal(coupling.v1, v1)
        np.testing.assert_almost_equal(coupling.u2, u2)
        np.testing.assert_almost_equal(coupling.v2, v2)

    def test_get_fourier_coefficients_case1(self):
        """ Check that the coefficient at the Gamma point is real and equal to one."""
        coupling = Coupling(**self.params)
        c_of_k, _ = coupling.get_fourier_coefficients(np.array([0, 0, 0]))
        self.assertEqual(np.imag(c_of_k), 0)
        self.assertEqual(np.real(c_of_k), 1)

    def test_get_fourier_coefficients_case2(self):
        """ Check that the coefficients are periodic."""
        coupling = Coupling(**self.params)
        k = np.array([0.1, 0.2, 0.3])
        c_of_k, _ = coupling.get_fourier_coefficients(k)
        c_of_k_plus_R, _ = coupling.get_fourier_coefficients(k + [-1, 2, 100])
        self.assertAlmostEqual(c_of_k, c_of_k_plus_R)

    def test_get_fourier_coefficients_case3(self):
        """ Check that the inner derivative of the coefficient is correct."""
        coupling = Coupling(**self.params)
        k = np.array([0.1, 0.2, 0.3])
        c_of_k, inner = coupling.get_fourier_coefficients(k)
        derivative = np.exp(-1j * (coupling.R @ k) * 2 * np.pi) * -1j * coupling.R * 2 * np.pi
        np.testing.assert_almost_equal(c_of_k * inner, derivative)

    def test_get_sw_matrix_elements_case1(self):
        """ Check that you cannot compute the matrix elements without having set the magnetic moments in the model."""
        coupling = Coupling(**self.params)
        with self.assertRaises(KeyError):
            coupling.get_sw_matrix_elements(np.array([0, 0, 0]))

    def test_get_sw_matrix_elements_case2(self):
        """ That the diagonal matrix elements are real."""
        setup_params = {'moment1': np.array([1, 0, 0]),
                        'moment2': np.array([1, -1, 1]),
                        'strength': -1,
                        'DM': [0.1, -0.2, 0.3]}
        coupling = self.set_moments_and_couplings(setup_params)
        k = np.array([-0.3, 0.4, 0.5])
        _, _, c_i, c_j, _, _, _ = coupling.get_sw_matrix_elements(k)
        self.assertAlmostEqual(np.imag(c_i), 0)
        self.assertAlmostEqual(np.imag(c_j), 0)

    def test_get_sw_matrix_elements_case3(self):
        """ Check that at the Gamma-point the a-type matrix elements are real in the FM case without DM."""
        setup_params = {'moment1': np.array([1, 1, 0]),
                        'moment2': np.array([1, 1, 0]),
                        'strength': -1,
                        'DM': [0, 0, 0]}
        coupling = self.set_moments_and_couplings(setup_params)
        k = np.array([0, 0, 0])
        a, a_bar, _, _, _, _, _ = coupling.get_sw_matrix_elements(k)
        self.assertEqual(np.imag(a), 0)
        self.assertEqual(np.imag(a_bar), 0)

    def test_get_sw_matrix_elements_case4(self):
        """ Check that at any k-point the a-type matrix elements are zero in the AFM case with DM."""
        setup_params = {'moment1': np.array([1, 1, 0]),
                        'moment2': np.array([-1, -1, 0]),
                        'strength': -1,
                        'DM': [0.2, 0.3, 0.2]}
        coupling = self.set_moments_and_couplings(setup_params)
        k = np.array([0.2, 10, -0.4])
        a, a_bar, _, _, _, _, _ = coupling.get_sw_matrix_elements(k)
        self.assertAlmostEqual(a, 0)
        self.assertAlmostEqual(a_bar, 0)

    def test_get_sw_matrix_elements_case5(self):
        """ Check that b-type matrix elements vanish in the FM case with arbitrary DM at any k-point."""
        setup_params = {'moment1': np.array([0, 1, 0]),
                        'moment2': np.array([0, 1, 0]),
                        'strength': -1,
                        'DM': [0.1, -0.2, 0.3]}
        coupling = self.set_moments_and_couplings(setup_params)
        k = np.array([0.3, -0.4, 10.2])
        _, _, _, _, b, b_bar, _ = coupling.get_sw_matrix_elements(k)
        self.assertEqual(b, 0)
        self.assertEqual(b_bar, 0)

    def test_get_sw_matrix_elements_case6(self):
        """ Check that at the Gamma-point the b-type matrix elements are complex conjugates in the AFM case with DM."""
        setup_params = {'moment1': np.array([1, 1, 0]),
                        'moment2': np.array([-1, -1, 0]),
                        'strength': -1,
                        'DM': [0., 0., 0.]}
        coupling = self.set_moments_and_couplings(setup_params)
        k = np.array([0, 0, 0])
        _, _, _, _, b, b_bar, _ = coupling.get_sw_matrix_elements(k)
        self.assertAlmostEqual(np.imag(b), 0)
        self.assertAlmostEqual(np.imag(b_bar), 0)


if __name__ == '__main__':
    unittest.main()
