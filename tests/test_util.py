import unittest

import numpy as np
from numpy.linalg import norm

from topwave import util

class UtilTest(unittest.TestCase):

    def test_rotate_vector_to_ez_case1(self):
        """ Check that the rotation matrix applied to the input vector is [0, 0, 1]."""
        v = [1, 0, 0]
        r = util.rotate_vector_to_ez(v)
        np.testing.assert_almost_equal(v @ r, [0, 0, 1])

    def test_rotate_vector_to_ez_case2(self):
        """ Check that the rotation matrix does not change the length of a non unit vector."""
        v = [1, 2, 3]
        r = util.rotate_vector_to_ez(v)
        self.assertAlmostEqual(norm(v @ r), norm(v))

    def test_rotate_vector_to_ez_case3(self):
        """ Check that a vector parallel to [0, 0, 1] yields the identity matrix."""
        v = [0, 0, 10]
        r = util.rotate_vector_to_ez(v)
        np.testing.assert_almost_equal(np.eye(3), r)


if __name__ == '__main__':
    unittest.main()
