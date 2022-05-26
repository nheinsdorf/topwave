import unittest

from topwave import util

import numpy as np


class UtilTest(unittest.TestCase):

    def test_rotate_vector_to_ez(self):
        v1 = [1, 0, 0]
        R1 = util.rotate_vector_to_ez(v1)
        np.testing.assert_almost_equal(v1 @ R1, [0, 0, 1])


if __name__ == '__main__':
    unittest.main()