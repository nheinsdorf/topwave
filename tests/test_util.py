from collections.abc import Iterable
import unittest

import numpy as np
from numpy.linalg import norm

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
    """I'll implement this checks when I have a test case of model."""
    def test_coupling_selector_case1(self):
        """Check that..."""
        pass


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
    """I'll implement this checks when I have a test case of model."""
    def test_get_boundary_couplings_case1(self):
        """Check that..."""
        pass


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
