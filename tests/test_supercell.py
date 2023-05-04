import unittest

import numpy as np
from pymatgen.core.structure import Structure

from topwave.model import SpinWaveModel, TightBindingModel
from topwave.supercell import Supercell


class SupercellTest1(unittest.TestCase):

    def setUp(self):
        """We use the non-chiral space group 198 with four magnetic sites as an example for testing."""

        self.space_group = 198
        self.max_distance = 7
        self.structure = Structure.from_spacegroup(self.space_group, 8.908 * np.eye(3), ['Cu'], [[0., 0., 0.]])
        self.model = SpinWaveModel(self.structure.copy())

    def test_init_case1(self):
        """Checks that the scaling factors are saved in the class instance."""

        scaling_factors = np.arange(3)
        supercell = Supercell(self.model, scaling_factors)
        np.testing.assert_equal(supercell.scaling_factors, scaling_factors)

    def test_init_case2(self):
        """Checks that the Zeeman term from a previous model is copied over."""

        zeeman = [22, 4, 1]
        self.model.set_zeeman(zeeman)
        supercell = Supercell(self.model, [2, 1, 1])
        np.testing.assert_equal(supercell.zeeman, zeeman)

    def test_get_supercell_structure_case1(self):
        """Checks that the supercell cannot be created if not exactly three scaling factors are provided."""

        with self.assertRaises(ValueError):
            supercell = Supercell(self.model, [1, 2])
        with self.assertRaises(ValueError):
            supercell = Supercell(self.model, (1, 2, 3, 4))

    def test_get_supercell_structure_case2(self):
        """Checks that none of the scaling factors can be zero."""

        pass

    def test_get_supercell_structure_case3(self):
        """Checks that the [1, 1, 1] supercell is the same as the original structure."""

        pass

    def test_get_type_case1(self):
        """Checks that the type for the model is returned correctly."""

        self.assertTrue(self.model.get_type() == 'spinwave')

class SupercellTest2(unittest.TestCase):

    def setUp(self):
        """We use the hmm model as an example for testing."""

        pass

    def test_get_type_case1(self):
        """Checks that the type for the model is returned correctly."""

        pass