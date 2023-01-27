from collections.abc import Iterable
import unittest

import numpy as np
from numpy.linalg import norm
from pymatgen.core.structure import Structure

from topwave.model import SpinWaveModel

from topwave.constants import PAULI_X, PAULI_Y, PAULI_Z, PAULI_VEC
from topwave import util


class BoseDistributionTest(unittest.TestCase):

    def test_bose_distribution_case1(self):
        """Check that zeroes are returned if temperature is zero."""

        occ = util.bose_distribution(np.linspace(0, 10, 51), temperature=0)
        np.testing.assert_almost_equal(occ, 0)

    def test_bose_distribution_case2(self):
        """Checks that high energies at moderate temperatures are unoccupied."""

        np.testing.assert_almost_equal(util.bose_distribution(800, temperature=100), 0)

    def test_bose_distribution_case3(self):
        """Checks if the return of the function is iterable even if the input is not."""

        self.assertTrue(isinstance(util.bose_distribution(0.2, temperature=10), Iterable))


class CouplingSelectorTest(unittest.TestCase):

    def setUp(self):
        """ We use the non-chiral space group 198 with four magnetic sites for testing."""

        self.space_group = 198
        self.max_distance = 7
        self.structure = Structure.from_spacegroup(self.space_group, 8.908 * np.eye(3), ['Cu'], [[0., 0., 0.]])
        self.model = SpinWaveModel(self.structure)

    def test_coupling_selector_case1(self):
        """Checks that no couplings are selected if more than a single value is passed."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        values = [0, 1]
        self.assertFalse(util.coupling_selector('is_set', values, self.model))
        self.assertFalse(util.coupling_selector('index', values, self.model))
        self.assertFalse(util.coupling_selector('symmetry_id', values, self.model))
        self.assertFalse(util.coupling_selector('distance', values, self.model))

    def test_coupling_selector_case2(self):
        """Checks the selection based on the 'is_set' attribute. (Also tested in test_model.py)."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        indices1 = [0, 4, 5, 6, 9, 14, 20, 21]
        indices2 = [0, 3, 4, 8, 11, 14]
        for index in indices1:
            self.model.set_coupling(index, 1.2)
        for index in indices2:
            self.model.set_spin_orbit(index, [1, 2, 3])
        indices = np.unique(np.concatenate([indices1, indices2], axis=0))
        np.testing.assert_equal(util.coupling_selector('is_set', True, self.model), indices)

    def test_coupling_selector_case3(self):
        """Checks the selection based on the 'index' attribute."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        indices = [0, 2, 9, 15, 23]
        for index in indices:
            self.assertEqual(util.coupling_selector('index', index, self.model)[0], index)

    def test_coupling_selector_case4(self):
        """Checks the selection based on symmetry index."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        symmetry_id = 1
        indices = [coupling.index for coupling in self.model.couplings if coupling.symmetry_id == symmetry_id]
        np.testing.assert_equal(util.coupling_selector('symmetry_id', symmetry_id, self.model), indices)

    def test_coupling_selector_case5(self):
        """Checks the selection based on distance."""

        self.model.generate_couplings(2 * self.max_distance, self.space_group)
        distance = self.model.couplings[0].distance
        indices1 = util.coupling_selector('distance', distance, self.model)
        self.assertLess(len(indices1), len(self.model.couplings))
        indices2 = [coupling.index for coupling in self.model.couplings if np.isclose(coupling.distance, distance)]
        np.testing.assert_equal(indices1, indices2)

    def test_coupling_selector_case6(self):
        """Checks that an empty list is returned when the attribute value is not matched."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        self.assertFalse(util.coupling_selector('is_set', True, self.model))
        self.assertFalse(util.coupling_selector('index', len(self.model.couplings) + 1, self.model))
        self.assertFalse(util.coupling_selector('symmetry_id', len(self.model.couplings) + 1, self.model))
        self.assertFalse(util.coupling_selector('distance', self.max_distance + 1, self.model))

    def test_coupling_selector_case7(self):
        """Checks that an error is raised if the attribute does not exist."""

        with self.assertRaises(UnboundLocalError):
            util.coupling_selector('unknown_attribute', 0, self.model)

    def test_coupling_selector_case8(self):
        """Checks that an empty list is returned when there are no couplings."""

        self.assertFalse(util.coupling_selector('index', 0, self.model))


class FormatInputVectorTest(unittest.TestCase):

    def test_format_input_vector_case1(self):
        """Check that a two-dimensional vector as input raises an error."""

        wrong_input = [2, 1]
        with self.assertRaises(ValueError):
            util.format_input_vector(wrong_input)

    def test_format_input_vector_case2(self):
        """Check that a nonunit vector is stretched properly."""

        length = 20
        vector = util.format_input_vector([1, 2, 3], length)
        self.assertAlmostEqual(norm(vector), length)

    def test_format_input_vector_case3(self):
        """Check that input is not stretched if length is None."""

        vector = [1, 2, 3]
        length = norm(vector)
        formatted_vector = util.format_input_vector(vector, length=None)
        self.assertAlmostEqual(norm(formatted_vector), length)


class GaussianTest(unittest.TestCase):

    def test_gaussian_case1(self):
        """Checks that at the mean value the prefactor is right."""

        std = 2
        pre_factor = 1 / (std * np.sqrt(2 * np.pi))
        self.assertAlmostEqual(util.gaussian(x=0, mean=0, std=std), pre_factor)

    def test_gaussian_case2(self):
        """Checks that the distribution integrates to one."""

        x_min, x_max, num_steps = 0, 1, 101
        xs = np.linspace(x_min, x_max, num_steps)
        distribution = util.gaussian(xs, mean=x_max / 2, std=0.05)
        dx = xs[1] - xs[0]
        integral = np.trapz(distribution, dx=dx)
        self.assertAlmostEqual(integral, 1)

    def test_gaussian_case5(self):
        """Check that eight standard deviations away from the mean it returns (close to) zero."""

        std = 0.2
        np.testing.assert_almost_equal(util.gaussian(8 * std, mean=0, std=std), 0.)

    def test_gaussian_case4(self):
        """Checks if the return of the function is iterable even if the input is not."""

        self.assertTrue(isinstance(util.gaussian(x=1, mean=0, std=2), Iterable))


class GetAzimuthalAngleTest(unittest.TestCase):

    def test_get_azimuthal_angle_case1(self):
        """Check that the angle w.r.t. to [1, 0, 0] is zero."""

        self.assertAlmostEqual(util.get_azimuthal_angle([1, 0, 0]), 0)

    def test_get_azimuthal_angle_case2(self):
        """Check the conversion to degree."""

        self.assertAlmostEqual(util.get_azimuthal_angle([0, 1, 0], deg=False), np.pi / 2)
        self.assertAlmostEqual(util.get_azimuthal_angle([0, 1, 0], deg=True), 90)

    def test_get_azimuthal_angle_case3(self):
        """Check that the length does not matter."""

        vector = np.array([0, 1, 1], dtype=np.float64)
        length = 10.
        self.assertAlmostEqual(util.get_azimuthal_angle(vector), util.get_azimuthal_angle(length * vector))


class GetBoundaryCouplingsTest(unittest.TestCase):

    def setUp(self):
        """ We use the non-chiral space group 198 with four magnetic sites for testing."""

        self.space_group = 198
        self.max_distance = 7
        self.structure = Structure.from_spacegroup(self.space_group, 8.908 * np.eye(3), ['Cu'], [[0., 0., 0.]])
        self.model = SpinWaveModel(self.structure)

    def test_get_boundary_couplings_case1(self):
        """Checks that there's the right amount of boundary couplings for the hoppings in each direction."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        num_x = len(util.get_boundary_couplings(self.model, 'x'))
        num_y = len(util.get_boundary_couplings(self.model, 'y'))
        num_z = len(util.get_boundary_couplings(self.model, 'z'))
        # there should be eight boundary hoppings in each direction.
        np.testing.assert_equal([num_x, num_y, num_z], [8] * 3)
        self.assertEqual(len(util.get_boundary_couplings(self.model, 'xy')), 14)
        self.assertEqual(len(util.get_boundary_couplings(self.model, 'yz')), 14)
        self.assertEqual(len(util.get_boundary_couplings(self.model, 'xz')), 14)
        self.assertEqual(len(util.get_boundary_couplings(self.model, 'xyz')), 18)

    def test_get_boundary_couplings_case2(self):
        """Checks that the order of the characters in the string does not matter."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        np.testing.assert_equal(util.get_boundary_couplings(self.model, 'xy'),
                                util.get_boundary_couplings(self.model, 'yx'))
        np.testing.assert_equal(util.get_boundary_couplings(self.model, 'xyz'),
                                util.get_boundary_couplings(self.model, 'zxy'))

    def test_get_boundary_couplings_case3(self):
        """Checks that also vectors that connect to a unit cell more than one lattice vector away are found."""

        self.model.generate_couplings(16.7, 1)
        indices_two_vectors = [coupling.index for coupling in self.model.couplings if np.abs(coupling.lattice_vector[0]) == 2]
        boundary_indices = util.get_boundary_couplings(self.model, 'x')
        for index in indices_two_vectors:
            self.assertIn(index, boundary_indices)


class GetElevationAngleTest(unittest.TestCase):

    def test_get_elevation_angle_case1(self):
        """Check that the angle w.r.t. to [0, 0, 1] is zero."""

        self.assertAlmostEqual(util.get_elevation_angle([0, 0, 1]), 0)

    def test_get_elevation_angle_case2(self):
        """Check the conversion to degree."""

        self.assertAlmostEqual(util.get_elevation_angle([0, 1, 0], deg=False), np.pi / 2)
        self.assertAlmostEqual(util.get_elevation_angle([0, 1, 0], deg=True), 90)

    def test_get_elevation_angle_case3(self):
        """Check that the length does not matter."""

        vector = np.array([1, 1, 0], dtype=np.float64)
        length = 10.
        self.assertAlmostEqual(util.get_elevation_angle(vector), util.get_elevation_angle(length * vector))

class PauliTest(unittest.TestCase):

    def test_pauli_case1(self):
        """Checks that the trace of the output is zero."""

        self.assertAlmostEqual(np.trace(util.pauli([2, 13, 5])), 0)

    def test_pauli_case2(self):
        """Checks that the unit vectors return the Pauli matrices."""

        np.testing.assert_almost_equal(util.pauli([1, 0, 0]), PAULI_X)
        np.testing.assert_almost_equal(util.pauli([0, 1, 0]), PAULI_Y)
        np.testing.assert_almost_equal(util.pauli([0, 0, 1]), PAULI_Z)

    def test_pauli_case3(self):
        """Checks the normalization."""

        vector = np.array([1, 1, 1], dtype=np.float64)
        np.testing.assert_almost_equal(util.pauli(vector, normalize=False), PAULI_VEC.sum(axis=0))
        np.testing.assert_almost_equal(util.pauli(vector, normalize=True), PAULI_VEC.sum(axis=0) / norm(vector))

    def test_pauli_case4(self):
        """Checks that an error is raised if the input vector is not three-dimensional."""

        wrong_input = np.arange(4)
        with self.assertRaises(ValueError):
            util.pauli(wrong_input)


class RotateVectorTest(unittest.TestCase):

    def test_rotate_vector_case1(self):
        """Check that its equivalent to the rotate_vector_to_ez-method."""

        vector = [1, 0, 0]
        rotated_vector = util.rotate_vector(vector, angle=-np.pi / 2, axis=[0, 1, 0])
        rotation_matrix =  util.rotate_vector_to_ez([1, 0, 0])
        np.testing.assert_almost_equal(rotated_vector, rotation_matrix.T @ vector)

    def test_rotate_vector_case2(self):
        """Check that the length is preserved."""

        vector = [20, 1, 39]
        rotated_vector = util.rotate_vector(vector, angle=2.41, axis=[10, 3, 1])
        self.assertAlmostEqual(norm(vector), norm(rotated_vector))

    def test_rotate_vector_case3(self):
        """Check that a 360 degree rotation does nothing."""

        vector = [22, 14, 3]
        rotated_vector = util.rotate_vector(vector, angle=2 * np.pi, axis=[0, 22, 3])
        np.testing.assert_almost_equal(vector, rotated_vector)

    def test_rotate_vector_case4(self):
        """Check that a 180 degree rotation about an orthogonal axis flips the sign."""

        vector = [4, 4, -5]
        rotated_vector = util.rotate_vector(vector, angle=np.pi, axis=np.cross(vector, [0, 0, 1]))
        np.testing.assert_almost_equal(vector, -rotated_vector)

    def test_rotate_vector_case5(self):
        """Checks that an error is raised if the input or axis vector is not three-dimensional."""

        wrong_input = np.arange(1)
        vector = [1, 2, 3]
        with self.assertRaises(ValueError):
            util.rotate_vector(wrong_input, angle=0.3, axis=vector)
        with self.assertRaises(ValueError):
            util.rotate_vector(vector, angle=0.3, axis=wrong_input)

    def test_rotate_vector_case6(self):
        """Checks SOMETHING WITH BASIS"""

        pass


class RotateVectorToEzTest(unittest.TestCase):

    def test_rotate_vector_to_ez_case1(self):
        """Check that the rotation matrix applied to the input vector is [0, 0, 1]."""

        v = [1, 0, 0]
        r = util.rotate_vector_to_ez(v)
        np.testing.assert_almost_equal(v @ r, [0, 0, 1])

    def test_rotate_vector_to_ez_case2(self):
        """Check that the rotation matrix does not change the length of a non unit vector."""

        v = [1, 2, 3]
        r = util.rotate_vector_to_ez(v)
        self.assertAlmostEqual(norm(v @ r), norm(v))

    def test_rotate_vector_to_ez_case3(self):
        """Check that a vector parallel to [0, 0, 1] yields the identity matrix."""

        v = [0, 0, 10]
        r = util.rotate_vector_to_ez(v)
        np.testing.assert_almost_equal(r, np.eye(3))

    def test_rotate_vector_to_ez_case4(self):
        """Checks that an error is raised if the input vector is not three-dimensional."""

        wrong_input = np.arange(5)
        with self.assertRaises(ValueError):
            util.rotate_vector_to_ez(wrong_input)


if __name__ == '__main__':
    unittest.main()
