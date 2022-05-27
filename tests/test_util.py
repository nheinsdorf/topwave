import unittest

from topwave import util

import numpy as np
from numpy.linalg import norm

class UtilTest(unittest.TestCase):

    def test_rotate_vector_to_ez_case1(self):
        v = [1, 0, 0]
        r = util.rotate_vector_to_ez(v)
        np.testing.assert_almost_equal(v @ r, [0, 0, 1])

    def test_rotate_vector_to_ez_case2(self):
        v = [1, 2, 3]
        r = util.rotate_vector_to_ez(v)
        self.assertAlmostEqual(norm(v @ r), norm(v))

    def test_rotate_vector_to_ez_case3(self):
        v = [0, 0, 10]
        r = util.rotate_vector_to_ez(v)
        np.testing.assert_almost_equal(np.eye(3), r)


if __name__ == '__main__':
    unittest.main()
