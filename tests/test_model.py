import io
import os
import sys
import unittest

import numpy as np
from numpy.linalg import norm
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup

from topwave.model import Model, SpinWaveModel, TightBindingModel
from topwave import util


class SpinWaveModelTest1(unittest.TestCase):

    def setUp(self):
        """We use the non-chiral space group 198 with four magnetic sites for testing."""

        self.space_group = 198
        self.max_distance = 7
        self.structure = Structure.from_spacegroup(self.space_group, 8.908 * np.eye(3), ['Cu'], [[0., 0., 0.]])
        self.model = SpinWaveModel(self.structure.copy())

    def test_init_case1(self):
        """Checks that the abstract parent class cannot be instantiated."""

        with self.assertRaises(TypeError):
            Model(self.structure)

    def test_init_case2(self):
        """Checks that there are no site properties when import_site_properties is true."""

        model = SpinWaveModel(self.structure, import_site_properties=True)
        print(self.structure[0].properties)
        for site in model.structure:
            self.assertDictEqual(site.properties, {})

    def test_init_case3(self):
        """Checks that the model starts without scaling factors, zeeman term or couplings."""

        self.assertIsNone(self.model.scaling_factors)
        np.testing.assert_almost_equal(self.model.zeeman, 0)
        self.assertFalse(self.model.couplings)

    def test_delete_all_couplings_case1(self):
        """Checks that all couplings are deleted."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        self.model.delete_all_couplings()
        self.assertFalse(self.model.couplings)

    def test_generate_couplings_case1(self):
        """Checks that calling the method again overrides old couplings."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        num_couplings1 = len(self.model.couplings)
        self.model.generate_couplings(self.max_distance, self.space_group)
        num_couplings2 = len(self.model.couplings)
        self.assertEqual(num_couplings1, num_couplings2)

    def test_generate_couplings_case2(self):
        """Checks that there are no symmetry indices when couplings are generated with P1 symmetry."""

        self.model.generate_couplings(self.max_distance, 1)
        indices = [coupling.index for coupling in self.model.couplings]
        symmetry_indices = [coupling.symmetry_id for coupling in self.model.couplings]
        np.testing.assert_equal(indices, symmetry_indices)

    def test_generate_couplings_case3(self):
        """ Check that the first occurrence of a symmetry index has the identity as operation."""

        for space_group in [1, self.space_group]:
            self.model.generate_couplings(self.max_distance, space_group)
            symmetry_indices = [coupling.symmetry_id for coupling in self.model.couplings]
            for symmetry_index in np.unique(symmetry_indices):
                index = np.argmin(np.abs(symmetry_indices - symmetry_index))
                operation = self.model.couplings[index].symmetry_op
                np.testing.assert_equal(operation.affine_matrix, np.eye(4))

    def test_generate_couplings_case4(self):
        """Checks that an error is raised when the space group is not a number between 1 and 230"""

        with self.assertRaises(TypeError):
            self.model.generate_couplings(self.max_distance, 0)
        with self.assertRaises(TypeError):
            self.model.generate_couplings(self.max_distance, -4)
        with self.assertRaises(TypeError):
            self.model.generate_couplings(self.max_distance, 231)

    def test_generate_couplings_case5(self):
        """Checks that no couplings are generated when max_dist is 0 or negative."""

        for distance in [0, -self.max_distance]:
            self.model.generate_couplings(distance, self.space_group)
            self.assertFalse(self.model.couplings)

    def test_generate_couplings_case6(self):
        """Checks that there are as many couplings as the pymatgen neighbor algorithm finds."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        num_couplings = len(self.model.couplings)
        num_neighbors = len(self.model.structure.get_symmetric_neighbor_list(self.max_distance, self.space_group, True)[0])
        self.assertEqual(num_couplings, num_neighbors)

    def test_generate_couplings_case7(self):
        """Checks that couplings of the same distance that are nonequivalent w.r.t. to symmetry are found."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        NN_distance = self.model.couplings[0].distance
        couplings = self.model.get_couplings('distance', NN_distance)
        symmetry_indices = [coupling.symmetry_id for coupling in couplings]
        num_symmetry_indices = len(np.unique(symmetry_indices))
        self.assertGreater(num_symmetry_indices, 1)

    def test_generate_couplings_case8(self):
        """Checks that all couplings are initialized unset and without strength or spin orbit interaction."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        for coupling in self.model.couplings:
            self.assertFalse(coupling.is_set)
            self.assertFalse(coupling.strength)
            np.testing.assert_equal(coupling.spin_orbit, np.zeros(3))

    def test_get_couplings_case1(self):
        """Checks that all set couplings are returned."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        indices1 = [0, 4, 2, 5, 18]
        indices2 = [4, 6, 20]
        for _ in indices1:
            self.model.set_coupling(_, 1)
        for _ in indices2:
            self.model.set_spin_orbit(_, [1, 2, 3])
        set_couplings = self.model.get_couplings('is_set', True)
        all_indices = np.unique(np.concatenate([indices1, indices2], axis=0))
        self.assertEqual(len(set_couplings), len(all_indices))
        for index, coupling in zip(all_indices, set_couplings):
            self.assertEqual(self.model.couplings[index], coupling)

    def test_get_couplings_case2(self):
        """Checks that the right coupling by index is returned."""

        index = 10
        self.model.generate_couplings(self.max_distance, self.space_group)
        couplings = self.model.get_couplings('index', index)
        self.assertEqual(len(couplings), 1)
        self.assertEqual(couplings[0], self.model.couplings[index])

    def test_get_couplings_case3(self):
        """Checks that an empty list is returned when a list is passed."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        couplings = self.model.get_couplings('index', [0, 2, 5])
        self.assertFalse(couplings)

    def test_get_couplings_case4(self):
        """Checks that all the right couplings based on symmetry index are returned."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        couplings1 = self.model.get_couplings('symmetry_id', 1)
        couplings2 = [coupling for coupling in self.model.couplings if coupling.symmetry_id == 1]
        self.assertEqual(len(couplings1), len(couplings2))
        for coupling1, coupling2 in zip(couplings1, couplings2):
            self.assertEqual(coupling1, coupling2)

    def test_get_couplings_case5(self):
        """Checks that all the right couplings based on distance index are returned."""

        self.model.generate_couplings(2 * self.max_distance, self.space_group)
        NN_distance = self.model.couplings[0].distance
        couplings1 = self.model.get_couplings('distance', NN_distance)
        couplings2 = [coupling for coupling in self.model.couplings if np.isclose(coupling.distance, NN_distance)]
        distances = np.unique([coupling.distance for coupling in self.model.couplings])
        self.assertGreater(len(distances), 1)
        self.assertEqual(len(couplings1), len(couplings2))
        for coupling1, coupling2 in zip(couplings1, couplings2):
            self.assertEqual(coupling1, coupling2)

    def test_get_set_couplings_case1(self):
        """Checks that only set couplings are returned. (It's just a wrapper of 'get_couplings'.)"""

        self.test_get_couplings_case1()

    def test_get_type_case1(self):
        """Checks that the right type is returned."""

        self.assertEqual(self.model.get_type(), 'spinwave')

    def test_invert_couplings_case1(self):
        """Checks that the direction of the coupling is inverted."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        index = 5
        coupling = self.model.couplings[index]
        lattice_vector = coupling.lattice_vector
        site1_index = coupling.site1.properties['index']
        site2_index = coupling.site2.properties['index']
        self.model.invert_coupling(index)
        inverted_coupling = self.model.couplings[index]
        np.testing.assert_equal(-inverted_coupling.lattice_vector, lattice_vector)
        self.assertEqual(inverted_coupling.site2.properties['index'], site1_index)
        self.assertEqual(inverted_coupling.site1.properties['index'], site2_index)
        self.assertEqual(inverted_coupling.distance, coupling.distance)
        np.testing.assert_almost_equal(-inverted_coupling.sublattice_vector, coupling.sublattice_vector)

    def test_set_coupling_case1(self):
        """Checks that the strength of coupling is saved and that is_set is set to true."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        index = 4
        strength = 2.2
        self.model.set_coupling(index, strength)
        self.assertEqual(self.model.couplings[index].strength, strength)
        self.assertTrue(self.model.couplings[index].is_set)

    def test_set_coupling_case2(self):
        """Checks that calling the method again overrides the old strength."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        index = 2
        strength1 = 10
        self.model.set_coupling(index, strength1)
        strength2 = 14
        self.model.set_coupling(index, strength2)
        self.assertNotEqual(self.model.couplings[index].strength, strength1)
        self.assertEqual(self.model.couplings[index].strength, strength2)

    def test_set_moments_case1(self):
        """Checks that you cannot pass non three-dimensional vectors."""

        wrong_moments = [[1, 0]] * len(self.model.structure)
        with self.assertRaises(ValueError):
            self.model.set_moments(wrong_moments)

    def test_set_moments_case2(self):
        """Checks that the moments are not normalized when magnitudes is None."""

        moments = np.array([[1, 2, 3]] * len(self.model.structure), dtype=np.float64)
        self.model.set_moments(moments)
        for moment, site in zip(moments, self.model.structure):
            np.testing.assert_almost_equal(site.properties['magmom'], moment)

    def test_set_moments_case3(self):
        """Checks that the moments are normalized."""

        moment = [1, 1, 0]
        normalized_moment = np.array(moment, dtype=np.float64) / norm(moment)
        moments = np.array([moment] * len(self.model.structure), dtype=np.float64)
        magnitudes = np.arange(10)
        self.model.set_moments(moments, magnitudes)
        for magnitude, site in zip(magnitudes, self.model.structure):
            np.testing.assert_almost_equal(site.properties['magmom'], normalized_moment * magnitude)

    def test_set_moments_case4(self):
        """Checks that each site has a rotation matrix after the moments are set."""

        num_sites = len(self.model.structure)
        moments = np.arange(num_sites * 3).reshape((num_sites, 3))
        magnitudes = [2] * num_sites
        self.model.set_moments(moments, magnitudes)
        for moment, site in zip(moments, self.model.structure):
            moment = moment / norm(moment)
            rotation = util.rotate_vector_to_ez(moment)
            np.testing.assert_almost_equal(site.properties['Rot'],rotation)
            np.testing.assert_almost_equal(rotation.T @ moment, np.array([0, 0, 1]))

    def test_set_moments_case5(self):
        """Check that you cannot pass the orientations with the right number of elements but wrong shape."""

        num_sites = len(self.model.structure)
        orientations = np.arange(num_sites * 3)
        magnitudes = np.arange(num_sites)
        with self.assertRaises(ValueError):
            self.model.set_moments(orientations)
        with self.assertRaises(ValueError):
            self.model.set_moments(orientations, magnitudes)

    def test_set_onsite_scalar_case1(self):
        """Checks that calling the method twice overrides old values."""

        index1 = 2
        strength1 = 3.4
        self.model.set_onsite_scalar(index1, strength1)
        self.assertEqual(self.model.structure[index1].properties['onsite_scalar'], strength1)
        index2 = 3
        self.model.set_onsite_scalar(index2, strength1)
        self.assertEqual(self.model.structure[index2].properties['onsite_scalar'], strength1)
        strength2 = 2.1
        self.model.set_onsite_scalar(index1, strength2)
        self.assertEqual(self.model.structure[index1].properties['onsite_scalar'], strength2)
        self.assertEqual(self.model.structure[index2].properties['onsite_scalar'], strength1)

    def test_set_onsite_scalar_case2(self):
        """Checks that the space group symmetry is used to assign the site properties."""

        strength = 3.2
        self.model.set_onsite_scalar(0, strength, space_group=self.space_group)
        for site in self.model.structure:
            self.assertEqual(site.properties['onsite_scalar'], strength)
        self.model.set_onsite_scalar(2, strength, space_group=self.space_group)
        for site in self.model.structure:
            self.assertEqual(site.properties['onsite_scalar'], strength)

    def test_set_onsite_scalar_case3(self):
        """Checks that setting the onsite scalar without a given space group only changes it on one site."""

        strength1 = 0.8
        index = 1
        self.model.set_onsite_scalar(index, strength1, space_group=self.space_group)
        strength2 = 1.4
        self.model.set_onsite_scalar(index, strength2)
        strengths = [strength1] * len(self.model.structure)
        strengths[index] = strength2
        for site, strength in zip(self.model.structure, strengths):
            self.assertEqual(site.properties['onsite_scalar'], strength)

    def test_set_onsite_scalar_case4(self):
        """Checks that you cannot pass multiple indices at once."""

        with self.assertRaises(TypeError):
            self.model.set_onsite_scalar([0, 1], 0.4)

    def test_set_onsite_scalar_case5(self):
        """Checks that you can only pass a strength that is convertible to a float."""

        with self.assertRaises(TypeError):
            self.model.set_onsite_scalar(0, [0, 4, 3])
        with self.assertRaises(TypeError):
            self.model.set_onsite_scalar(1, np.arange(3))

    def test_set_onsite_scalar_case6(self):
        """Checks that values are assigned by symmetry correctly if there are atoms on different Wyckoff positions."""

        structure = Structure.from_spacegroup(198, np.eye(3), ['Cu', 'O'], [[0, 0, 0], [0.1, 0, 0]])
        model = SpinWaveModel(structure)
        strength1 = 1.
        strength2 = 2.
        model.set_onsite_scalar(0, strength1, space_group=self.space_group)
        model.set_onsite_scalar(4, strength2, space_group=self.space_group)
        for site in model.structure:
            if site.species_string == 'Cu':
                self.assertEqual(site.properties['onsite_scalar'], strength1)
            else:
                self.assertEqual(site.properties['onsite_scalar'], strength2)

    def test_set_onsite_vector_case1(self):
        """Checks that calling the method twice overrides old values."""

        index1 = 2
        vector1 = [1, 2, 3]
        self.model.set_onsite_vector(index1, vector1)
        np.testing.assert_equal(self.model.structure[index1].properties['onsite_vector'], vector1)
        index2 = 3
        self.model.set_onsite_vector(index2, vector1)
        np.testing.assert_equal(self.model.structure[index2].properties['onsite_vector'], vector1)
        strength2 = [2, 3, 4]
        self.model.set_onsite_vector(index1, strength2)
        np.testing.assert_equal(self.model.structure[index1].properties['onsite_vector'], strength2)
        np.testing.assert_equal(self.model.structure[index2].properties['onsite_vector'], vector1)

    def test_set_onsite_vector_case2(self):
        """Checks that the selected site and not the zeroth site in the structure is assigned the vector."""

        vector = [4, 4, 4]
        index = 3
        self.model.set_onsite_vector(index, vector, space_group=self.space_group)
        np.testing.assert_equal(self.model.structure[index].properties['onsite_vector'], vector)

    def test_set_onsite_vector_case3(self):
        """Checks that the space group symmetry is used to rotate the vectors."""

        vector = [3, 2, 3]
        index = 2
        self.model.set_onsite_vector(index, vector, space_group=self.space_group)

        space_group = SpaceGroup.from_int_number(self.space_group)
        coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
        for coordinate, operation in zip(coordinates, operations):
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.model.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            np.testing.assert_almost_equal(site.properties['onsite_vector'], operation.apply_rotation_only(vector))

    def test_set_onsite_vector_case4(self):
        """Checks that a non three-dimensional vector raises an error."""

        with self.assertRaises(ValueError):
            self.model.set_onsite_vector(0, [1, 2])

    def test_set_onsite_vector_case5(self):
        """Checks that you can only pass one index."""

        with self.assertRaises(TypeError):
            self.model.set_onsite_vector([1, 2], [0, 0, 1])
        with self.assertRaises(TypeError):
            self.model.set_onsite_vector([1, 2], [0, 0, 1], space_group=self.space_group)

    def test_set_onsite_vector_case6(self):
        """Checks that the inverted operation applied to the generated vectors yields the input vector."""

        vector = [1, 1, 3]
        index = 3
        self.model.set_onsite_vector(index, vector, space_group=self.space_group)

        space_group = SpaceGroup.from_int_number(self.space_group)
        coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
        for coordinate, operation in zip(coordinates, operations):
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.model.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            inverted_vector = operation.inverse.apply_rotation_only(site.properties['onsite_vector'])
            np.testing.assert_almost_equal(inverted_vector, vector)

    def test_set_open_boundaries_case1(self):
        """Checks that the selected boundaries (tested in test_util.py) are set to zero."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        strength = 2.3
        vector = [0, 0, 3]
        self.model.set_coupling(0, strength, 'symmetry_id')
        self.model.set_coupling(1, strength, 'symmetry_id')
        self.model.set_spin_orbit(0, [0, 0, 3], strength, 'symmetry_id')
        self.model.set_spin_orbit(1, [0, 0, 3], strength, 'symmetry_id')
        self.model.set_open_boundaries('x')
        indices = util.get_boundary_couplings(self.model, 'x')
        for coupling in self.model.couplings:
            if coupling.index in indices:
                self.assertFalse(coupling.strength)
                np.testing.assert_equal(coupling.spin_orbit, np.zeros(3))
                self.assertFalse(coupling.is_set)
            else:
                self.assertEqual(coupling.strength, 2.3)
                self.assertTrue(coupling.spin_orbit.any())
                self.assertTrue(coupling.is_set)

    def test_set_spin_orbit_case1(self):
        """Checks that the spin orbit term is saved for the right coupling and that 'is_set' is set to True."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        vector = [4, 2, 1]
        index = 4
        self.model.set_spin_orbit(index, vector)
        np.testing.assert_equal(self.model.couplings[index].spin_orbit, vector)
        self.assertTrue(self.model.couplings[index].is_set)

    def test_spin_orbit_case2(self):
        """Checks that non three dimensional vectors cannot be passed."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        with self.assertRaises(ValueError):
            self.model.set_spin_orbit(0, np.arange(5))

    def test_spin_orbit_case3(self):
        """Checks that nothing happens when a non existing index is passed."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        self.model.set_spin_orbit(len(self.model.couplings) + 1, [1, 2, 3])
        for coupling in self.model.couplings:
            self.assertFalse(coupling.is_set)

    def test_spin_orbit_case4(self):
        """Checks that vectors are properly normalized."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        vector = np.array([1, 0, 1], dtype=np.float64)
        normalized_vector = vector / norm(vector)
        strength = 4
        index = 10
        self.model.set_spin_orbit(index, vector, strength, 'index')
        np.testing.assert_almost_equal(self.model.couplings[index].spin_orbit, normalized_vector * strength)

    def test_spin_orbit_case5(self):
        """Checks that vectors are rotated properly."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        vector = np.arange(3)
        index = 0
        self.model.set_spin_orbit(index, vector, attribute='symmetry_id')
        for coupling in self.model.get_couplings(index, 'symmetry_id'):
            np.testing.assert_almost_equal(coupling.spin_orbit, coupling.symmetry_op.apply_rotation_only(vector))

    def test_spin_orbit_case6(self):
        """Checks that the inverse operation on the rotated vectors yields the input vector."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        vector = [2, 1, 4]
        index = 1
        self.model.set_spin_orbit(index, vector, attribute='symmetry_id')
        for coupling in self.model.get_couplings(index, 'symmetry_id'):
            inverted_vector = coupling.symmetry_op.inverse.apply_rotation_only(coupling.spin_orbit)
            np.testing.assert_almost_equal(inverted_vector, vector)

    def test_set_zeeman_case1(self):
        """Checks that a new model has no Zeeman term."""

        np.testing.assert_equal(self.model.zeeman, np.zeros(3))

    def test_set_zeeman_case2(self):
        """Checks that the Zeeman term is saved."""

        vector = [5, 23, 45]
        self.model.set_zeeman(vector)
        np.testing.assert_equal(self.model.zeeman, vector)

    def test_set_zeeman_case3(self):
        """Checks that the vector is normalized properly."""

        vector = [0, 1, 1]
        strength = 1
        self.model.set_zeeman(vector, strength)
        np.testing.assert_almost_equal(self.model.zeeman, [0, 1 / np.sqrt(2), 1 / np.sqrt(2)])

    def test_show_couplings_case1(self):
        """Checks that no error is raised for generated couplings and no couplings and something is printed."""

        captured_output1 = io.StringIO()
        sys.stdout = captured_output1
        self.model.show_couplings()
        self.assertFalse(captured_output1.getvalue() == '')

        captured_output2 = io.StringIO()
        sys.stdout = captured_output2
        self.model.generate_couplings(self.max_distance, self.space_group)
        self.model.show_couplings()
        self.assertFalse(captured_output2.getvalue() == '')
        sys.stdout = sys.__stdout__

    def test_show_site_properties_case1(self):
        """Checks that something is printed and no error is raised."""

        captured_output = io.StringIO()
        sys.stdout = captured_output
        self.model.show_site_properties()
        self.assertFalse(captured_output.getvalue() == '')

    def test_unset_coupling_case1(self):
        """Checks that a selected coupling that has been set using 'set_coupling' is unset."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        index = 12
        self.model.set_coupling(index, 2)
        self.model.unset_coupling(index)
        self.assertFalse(self.model.couplings[index].strength)
        self.assertFalse(self.model.couplings[index].is_set)

    def test_unset_coupling_case2(self):
        """Checks that selected couplings (by distance) that have been set using 'set_spin_orbit' are unset."""

        self.model.generate_couplings(self.max_distance, self.space_group)
        vector = [2, 31, 2]
        self.model.set_spin_orbit(0, vector, attribute='symmetry_id')
        self.model.set_spin_orbit(1, vector, attribute='symmetry_id')
        self.model.unset_coupling(self.model.couplings[0].distance, 'distance')
        for coupling in self.model.couplings:
            np.testing.assert_equal(coupling.spin_orbit, np.zeros(3))
            self.assertFalse(coupling.is_set)

    def test_unset_moments_case1(self):
        """Checks that all moments are unset."""

        num_sites = len(self.model.structure)
        self.model.set_moments([[1, 2, 3]] * num_sites)
        self.model.unset_moments()
        for site in self.model.structure:
            self.assertIsNone(site.properties['magmom'])

    def test_write_cif_case1(self):
        """Checks that a file is generated."""

        path = os.getcwd() + 'test.cif'
        self.model.write_cif(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        self.assertFalse(os.path.exists(path))

    def test_get_classical_energy_case1(self):
        """Checks that we have the correct ground state energy."""

        pass

    def test_set_single_ion_anisotropy_case1(self):
        """Checks that calling the method twice overrides old values."""

        index1 = 2
        vector1 = [1, 2, 3]
        self.model.set_single_ion_anisotropy(index1, vector1)
        np.testing.assert_equal(self.model.structure[index1].properties['onsite_vector'], vector1)
        index2 = 3
        self.model.set_single_ion_anisotropy(index2, vector1)
        np.testing.assert_equal(self.model.structure[index2].properties['onsite_vector'], vector1)
        strength2 = [2, 3, 4]
        self.model.set_single_ion_anisotropy(index1, strength2)
        np.testing.assert_equal(self.model.structure[index1].properties['onsite_vector'], strength2)
        np.testing.assert_equal(self.model.structure[index2].properties['onsite_vector'], vector1)

    def test_set_single_ion_anisotropy_case2(self):
        """Checks that the selected site and not the zeroth site in the structure is assigned the vector."""

        vector = [4, 4, 4]
        index = 3
        self.model.set_single_ion_anisotropy(index, vector, space_group=self.space_group)
        np.testing.assert_equal(self.model.structure[index].properties['onsite_vector'], vector)

    def test_set_single_ion_anisotropy_case3(self):
        """Checks that the space group symmetry is used to rotate the vectors."""

        vector = [3, 2, 3]
        index = 2
        self.model.set_single_ion_anisotropy(index, vector, space_group=self.space_group)

        space_group = SpaceGroup.from_int_number(self.space_group)
        coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
        for coordinate, operation in zip(coordinates, operations):
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.model.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            np.testing.assert_almost_equal(site.properties['onsite_vector'], operation.apply_rotation_only(vector))

    def test_set_single_ion_anisotropy_case4(self):
        """Checks that a non three-dimensional vector raises an error."""

        with self.assertRaises(ValueError):
            self.model.set_single_ion_anisotropy(0, [1, 2])

    def test_set_single_ion_anisotropy_case5(self):
        """Checks that you can only pass one index."""

        with self.assertRaises(TypeError):
            self.model.set_single_ion_anisotropy([1, 2], [0, 0, 1])
        with self.assertRaises(TypeError):
            self.model.set_single_ion_anisotropy([1, 2], [0, 0, 1], space_group=self.space_group)

    def test_set_single_ion_anisotropy_case6(self):
        """Checks that the inverted operation applied to the generated vectors yields the input vector."""

        vector = [1, 1, 3]
        index = 3
        self.model.set_single_ion_anisotropy(index, vector, space_group=self.space_group)

        space_group = SpaceGroup.from_int_number(self.space_group)
        coordinates, operations = space_group.get_orbit_and_generators(self.structure[index].frac_coords)
        for coordinate, operation in zip(coordinates, operations):
            cartesian_coordinate = self.structure.lattice.get_cartesian_coords(coordinate)
            site = self.model.structure.get_sites_in_sphere(cartesian_coordinate, 1e-06)[0]
            inverted_vector = operation.inverse.apply_rotation_only(site.properties['onsite_vector'])
            np.testing.assert_almost_equal(inverted_vector, vector)

class TightBindingModelTest1(unittest.TestCase):

    def setUp(self):
        """We use the hmm, model as an example to test this class."""

    def test_check_if_spinful_case1(self):
        """Checks that..."""
        pass

    def test_get_type_case1(self):
        """Checks that the correct type is returned."""
        pass


if __name__ == '__main__':
    unittest.main()
