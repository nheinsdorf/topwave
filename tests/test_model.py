import unittest

from topwave import model

import numpy as np
from numpy.linalg import norm
from pymatgen.core.structure import Structure


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




if __name__ == '__main__':
    unittest.main()
