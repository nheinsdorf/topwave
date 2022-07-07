import unittest

import numpy as np
from numpy.linalg import norm
from pymatgen.core.structure import Structure

from topwave import model

class ModelTest(unittest.TestCase):
    def setUp(self):
        """ We will use the non-chiral space group 198 with four magnetic sites for testing."""
        struc = Structure.from_spacegroup(198, 8.908 * np.eye(3), ['Cu'], [[0., 0., 0.]])
        self.model = model.Model(struc)
        self.max_dist = 7

    def test_generate_couplings_case1(self):
        """ Check that couplings are reset in list and dataframe when method is called again."""
        self.model.generate_couplings(self.max_dist, 198)
        num_couplings = len(self.model.CPLS)
        len_df = len(self.model.CPLS_as_df.index)
        self.assertEqual(num_couplings, len_df)
        self.model.generate_couplings(self.max_dist, 198)
        self.assertEqual(num_couplings, len(self.model.CPLS))
        self.assertEqual(len_df, len(self.model.CPLS_as_df.index))

    def test_generate_couplings_case2(self):
        """ Check that data in the list and the dataframe are equivalent."""
        self.model.generate_couplings(self.max_dist, 198)
        df = self.model.CPLS_as_df
        for _, coupling in enumerate(self.model.CPLS):
            self.assertEqual(coupling.ID, df.index[_])
            self.assertEqual(coupling.SYMID, df['symid'][_])
            self.assertEqual(coupling.SYMOP.as_xyz_string(), df['symop'][_])
            self.assertEqual(coupling.I, df['i'][_])
            self.assertEqual(coupling.J, df['j'][_])
            self.assertEqual(str(coupling.SITE1.species), df['at1'][_])
            self.assertEqual(str(coupling.SITE2.species), df['at2'][_])
            np.testing.assert_almost_equal(coupling.DELTA, df['delta'][_])
            np.testing.assert_equal(coupling.R, df['R'][_])
            self.assertAlmostEqual(coupling.D, df['dist'][_])
            self.assertEqual(coupling.strength, df['strength'][_])
            np.testing.assert_equal(coupling.DM, df['DM'][_])

    def test_generate_couplings_case3(self):
        """ Check that there are no symmetry groups when couplings are generated with P1 symmetry."""
        self.model.generate_couplings(self.max_dist, 1)
        indices = np.array(self.model.CPLS_as_df.index.to_list(), dtype=int)
        sym_indices = np.array(self.model.CPLS_as_df['symid'].to_list(), dtype=int)
        np.testing.assert_equal(indices, sym_indices)

    def test_generate_couplings_case4(self):
        """ Check that the first occurrence of a symmetry index has the identity as operation."""
        for space_group in [1, 198]:
            self.model.generate_couplings(self.max_dist, space_group)
            df = self.model.CPLS_as_df
            symmetry_indices = np.unique(df['symid'].to_list())
            for symmetry_index in symmetry_indices:
                index = df[df['symid'] == int(symmetry_index)].index.to_list()[0]
                operation = self.model.CPLS[index].SYMOP
                np.testing.assert_equal(operation.affine_matrix, np.eye(4))

    def test_generate_couplings_case5(self):
        """ Check that no couplings are generated when max_dist is 0 or negative."""
        for distance in [0, -self.max_dist]:
            self.model.generate_couplings(distance, 198)
            self.assertFalse(len(self.model.CPLS))
            self.assertFalse(len(self.model.CPLS_as_df))

    def test_generate_couplings_case6(self):
        """ Check that u and v are computed when the moments are set before the couplings are generated. """
        self.model.set_moments(self.model.N * [[0, 0, 1]], self.model.N * [0.5])
        self.model.generate_couplings(self.max_dist, 198)
        for coupling in self.model.CPLS:
            self.assertIsNotNone(coupling.u1)
            self.assertIsNotNone(coupling.u2)
            self.assertIsNotNone(coupling.v1)
            self.assertIsNotNone(coupling.v2)

    def test_set_couplings_case1(self):
        """ Check that the coupling strength of a coupling is initialized as 0. """
        self.model.generate_couplings(self.max_dist, 198)
        self.assertTrue(all(self.model.CPLS[_].strength == 0 for _ in range(len(self.model.CPLS))))

    def test_set_couplings_case2(self):
        """ Check that you cannot pass a list of indices to set strengths."""
        self.model.generate_couplings(self.max_dist, 198)
        with self.assertRaises(Exception):
            self.model.set_coupling(strength=1, index=[0, 1], by_symmetry=True)
        with self.assertRaises(Exception):
            self.model.set_coupling(strength=1, index=[0, 1], by_symmetry=False)
        # Also check that all strengths are still zero.
        self.assertTrue(all(self.model.CPLS[_].strength == 0 for _ in range(len(self.model.CPLS))))

    def test_set_couplings_case3(self):
        """ Check that no strengths are assigned if a nonexistent index or symmetry index is given."""
        self.model.generate_couplings(self.max_dist, 198)
        index = np.max(self.model.CPLS_as_df['symid']) + 1
        self.model.set_coupling(strength=1, index=index, by_symmetry=True)
        self.assertTrue(all(self.model.CPLS[_].strength == 0 for _ in range(len(self.model.CPLS))))
        self.model.set_coupling(strength=1, index=-index, by_symmetry=True)
        self.assertTrue(all(self.model.CPLS[_].strength == 0 for _ in range(len(self.model.CPLS))))
        index = np.max(self.model.CPLS_as_df.index) + 1
        self.model.set_coupling(strength=1, index=index, by_symmetry=False)
        self.assertTrue(all(self.model.CPLS[_].strength == 0 for _ in range(len(self.model.CPLS))))
        self.model.set_coupling(strength=1, index=-index, by_symmetry=False)
        self.assertTrue(all(self.model.CPLS[_].strength == 0 for _ in range(len(self.model.CPLS))))

    def test_set_field_case1(self):
        """ Check that you cannot give a direction that is not three-dimensional."""
        with self.assertRaises(Exception):
            self.model.set_field(1, 1)
        with self.assertRaises(Exception):
            self.model.set_field([1, 1], 1)
        with self.assertRaises(Exception):
            self.model.set_field([1, 1, 1, 1], 1)

    def test_set_field_case2(self):
        """ Check that the resulting numpy.ndarray is a float array even when magnitude is complex."""
        self.model.set_field([1, 1, 1], 1 + 1j)
        self.assertTrue(self.model.MF.dtype == float)

    def test_set_field_case3(self):
        """ Check that the direction was normalized."""
        magnitude = 3.841
        self.model.set_field([2, 1, 1], magnitude)
        self.assertAlmostEqual(norm(self.model.MF), magnitude)

    def test_set_moments_case1(self):
        """ Check that you cannot give non three-dimensional directions to the method."""
        with self.assertRaises(Exception):
            self.model.set_moments(self.model.N * [[0, 1]], self.model.N * [0.5])

    def test_set_moments_case2(self):
        """ Check that you cannot the wrong number of moments or directions to the method."""
        with self.assertRaises(Exception):
            self.model.set_moments((self.model.N - 1) * [[0, 0, 1]], (self.model.N - 1) * [0.5])

    def test_set_moments_case3(self):
        """ Check that the u and v vectors for the couplings are recomputed if the moments are changed."""
        self.model.generate_couplings(self.max_dist, 198)
        self.model.set_moments(self.model.N * [[0, 0, 1]], self.model.N * [0.5])
        uv = [(coupling.u1, coupling.u2, coupling.v1, coupling.v2) for coupling in self.model.CPLS]
        self.model.set_moments(self.model.N * [[1, 1, 1]], self.model.N * [0.5])
        for (coupling, uv) in zip(self.model.CPLS, uv):
            np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                     coupling.u1, uv[0])
            np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                     coupling.u2, uv[1])
            np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                     coupling.v1, uv[2])
            np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                     coupling.v2, uv[3])


class SpinWaveModelTest(unittest.TestCase):
    def setUp(self):
        """ We will use the non-chiral space group 198 with four ferromagnetic Spin=1 sites."""
        sg = 198
        struc = Structure.from_spacegroup(sg, 8.908 * np.eye(3), ['Cu'], [[0., 0., 0.]])
        self.model = model.SpinWaveModel(struc)
        self.max_dist = 7
        self.model.generate_couplings(7, sg)
        self.model.set_moments(self.model.N * [[0, 0, 1]], self.model.N * [1])

    def test_set_DM_case1(self):
        """ Check that applying the inverse rotation yields the original vector."""
        D = [0.1, 0.2, 0.3]
        symmetry_indices = np.unique(self.model.CPLS_as_df['symid'].to_list())
        for symmetry_index in symmetry_indices:
            self.model.set_DM(D, symmetry_index)
        for coupling in self.model.CPLS:
            np.testing.assert_almost_equal(coupling.SYMOP.inverse.apply_rotation_only(coupling.DM), D)


if __name__ == '__main__':
    unittest.main()
