"""
Set of regression tests for the geometry tools.
"""
import unittest

import numpy as np

from ase import Atoms
from ase.build import bcc100, molecule
from ase.visualize import view
import ase.build
import ase.lattice.hexagonal
import ase.io

import matid.geometry


class GeometryTests(unittest.TestCase):
    """Tests for the geometry module.
    """
    def test_get_nearest_atom(self):
        """Getting the nearest atom in a system.
        """
        # Test with a finite system
        system = molecule("H2O")
        system.set_cell(5*np.eye(3))
        system.center()

        location = np.array([2.5, 1.5, 6])
        index = matid.geometry.get_nearest_atom(system, location, mic=False)
        self.assertEqual(index, 0)  # Oxygen is closest without mic

        # Test with a periodic system
        system.set_pbc([True, True, True])
        index = matid.geometry.get_nearest_atom(system, location, mic=True)
        self.assertEqual(index, 2)  # Hydrogen is closest with mic

        # Test with a periodic system and a shifted position
        location2 = np.array([2.5, 3, 6])
        index = matid.geometry.get_nearest_atom(system, location2, mic=True)
        self.assertEqual(index, 1)  # Hydrogen is closest with mic

    def test_thickness(self):
        """Getting the thickness of structures.
        """
        system = molecule("H2O")
        system.set_cell(np.eye(3))
        thickness_x = matid.geometry.get_thickness(system, 0)
        self.assertEqual(thickness_x, 0)

        thickness_y = matid.geometry.get_thickness(system, 1)
        self.assertEqual(thickness_y, 1.526478)

        thickness_z = matid.geometry.get_thickness(system, 2)
        self.assertEqual(thickness_z, 0.596309)

    def test_minimize_cell(self):
        """Cell minimization.
        """
        system = molecule("H2O")
        system.set_cell([3, 3, 3])

        # Minimize in x-direction
        x_minimized_system = matid.geometry.get_minimized_cell(system, 0, 0.1)
        x_cell = x_minimized_system.get_cell()
        x_expected_cell = np.array([
            [0.1, 0., 0.],
            [0., 3., 0.],
            [0., 0., 3.]
        ])
        self.assertTrue(np.allclose(x_expected_cell, x_cell, atol=0.001, rtol=0))

        # Minimize in y-direction
        y_minimized_system = matid.geometry.get_minimized_cell(system, 1, 0.1)
        y_cell = y_minimized_system.get_cell()
        y_expected_cell = np.array([
            [3., 0., 0.],
            [0., 1.526478, 0.],
            [0., 0., 3.]
        ])
        self.assertTrue(np.allclose(y_expected_cell, y_cell, atol=0.001, rtol=0))

        # Minimize in z-direction with minimum size smaller than found minimum size
        minimized_system = matid.geometry.get_minimized_cell(system, 2, 0.1)
        cell = minimized_system.get_cell()
        pos = minimized_system.get_scaled_positions()
        expected_cell = np.array([
            [3., 0., 0.],
            [0., 3., 0.],
            [0., 0., 0.596309]
        ])
        expected_pos = np.array([
            [0., 0., 1.],
            [0., 0.254413, 0.],
            [0., -0.254413, 0.]
        ])
        self.assertTrue(np.allclose(expected_cell, cell, atol=0.001, rtol=0))
        self.assertTrue(np.allclose(expected_pos, pos, atol=0.001, rtol=0))

        # Minimize in z-direction with minimum size larger than found minimum size
        minimized_system = matid.geometry.get_minimized_cell(system, 2, 2)
        cell = minimized_system.get_cell()
        pos = minimized_system.get_scaled_positions()
        expected_cell = np.array([
            [3., 0., 0.],
            [0., 3., 0.],
            [0., 0., 2.]
        ])
        expected_pos = np.array([
            [0., 0., 0.64907725],
            [0., 0.254413, 0.35092275],
            [0., -0.254413, 0.35092275]
        ])
        self.assertTrue(np.allclose(expected_cell, cell, atol=0.001, rtol=0))
        self.assertTrue(np.allclose(expected_pos, pos, atol=0.001, rtol=0))

    def test_center_of_mass(self):
        """Tests that the center of mass correctly takes periodicity into
        account.
        """
        system = bcc100('Fe', size=(3, 3, 4), vacuum=8)
        adsorbate = ase.Atom(position=[4, 4, 4], symbol="H")
        system += adsorbate
        system.set_pbc([True, True, True])
        system.translate([0, 0, 10])
        system.wrap()

        # Test periodic COM
        cm = matid.geometry.get_center_of_mass(system)
        self.assertTrue(np.allclose(cm, [4., 4., 20.15], atol=0.1))

        # Test finite COM
        system.set_pbc(False)
        cm = matid.geometry.get_center_of_mass(system)
        self.assertTrue(np.allclose(cm, [3.58770672, 3.58770672, 10.00200455], atol=0.1))

        # Test positions with non-orthorhombic cell that has negative elements
        # in lattice vectors.
        system = Atoms(
            symbols=['N', 'N', 'N', 'N', 'C', 'C', 'N', 'C', 'H', 'C', 'C', 'N', 'C', 'H'],
            positions=[
                [5.99520154, 4.36260352, 9.34466641],
                [3.93237808, 3.20844954, 9.59369566],
                [-0.44330457, 4.00755999, 9.58355994],
                [1.61974674, 2.84737344, 9.36529391],
                [2.67745295, 3.5184896, 9.09252221],
                [4.71467838, 4.12389331, 9.05494284],
                [2.65906057, 4.65710632, 8.18163931],
                [3.88950688, 5.03564733, 8.17378237],
                [4.26274113, 5.88360129, 7.59608619],
                [0.33924086, 3.07840333, 9.06923781],
                [-0.48556248, 2.14395639, 8.21180484],
                [7.03506045, 2.52272554, 8.20901382],
                [7.05303326, 3.68462936, 9.08992327],
                [-0.11207002, 1.28102047, 7.65690464],
            ],
            cell=[
                [8.751006887253668, 0.0, 0.0],
                [-4.372606804779377, 11.962358903815558, 0.0],
                [0.00037467328230266297, -0.000720566930414705, 16.891557236170406]
            ],
            pbc=True
        )
        cm = matid.geometry.get_center_of_mass(system)
        self.assertTrue(np.allclose(cm, [6.2609094, 3.59987973, 8.90948045], atol=0.1))

    def test_matches_non_orthogonal(self):
        """Test that the correct factor is returned when finding matches that
        are in the neighbouring cells.
        """
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(5, 5, 1),
            vacuum=8)
        system.set_pbc(True)
        system = system[[0, 12]]
        # view(system)

        searched_pos = system.get_positions()[0][None, :]
        basis = np.array([[1.59, -2.75396078, 0]])
        searched_pos += basis

        matches, subst, vac, factors = matid.geometry.get_matches(
            system,
            searched_pos,
            numbers=[system.get_atomic_numbers()[0]],
            tolerances=np.array([0.2])
        )

        # Make sure that the atom is found in the correct copy
        self.assertEqual(tuple(factors[0]), (0, -1, 0))

        # Make sure that the correct atom is found
        self.assertTrue(np.array_equal(matches, [1]))

    def test_displacement_non_orthogonal(self):
        """Test that the correct displacement is returned when the cell in
        non-orthorhombic.
        """
        positions = np.array([
            [1.56909, 2.71871, 6.45326],
            [3.9248, 4.07536, 6.45326]
        ])
        cell = np.array([
            [4.7077, -2.718, 0.],
            [0., 8.15225, 0.],
            [0., 0., 50.]
        ])

        # Fully periodic with minimum image convention
        dist_mat = matid.geometry.get_distance_matrix(
            positions[0, :],
            positions[1, :],
            cell,
            pbc=True,
            mic=True)

        # The minimum image should be within the same cell
        expected = np.linalg.norm(positions[0, :] - positions[1, :])
        self.assertTrue(np.allclose(dist_mat[0], expected))

    def test_distance_matrix(self):
        pos1 = np.array([
            [0, 0, 0],
        ])
        pos2 = np.array([
            [0, 0, 7],
            [6, 0, 0],
        ])
        cell = np.array([
            [7, 0, 0],
            [0, 7, 0],
            [0, 0, 7]
        ])

        # Non-periodic
        dist_mat = matid.geometry.get_distance_matrix(pos1, pos2)
        expected = np.array(
            [[7, 6]]
        )
        self.assertTrue(np.allclose(dist_mat, expected))

        # Fully periodic with minimum image convention
        dist_mat = matid.geometry.get_distance_matrix(pos1, pos2, cell, pbc=True, mic=True)
        expected = np.array(
            [[0, 1]]
        )
        self.assertTrue(np.allclose(dist_mat, expected))

        # Partly periodic with minimum image convention
        dist_mat = matid.geometry.get_distance_matrix(pos1, pos2, cell, pbc=[False, True, True], mic=True)
        expected = np.array(
            [[0, 6]]
        )
        self.assertTrue(np.allclose(dist_mat, expected))

    def test_displacement_tensor(self):
        # Non-periodic
        cell = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pos1 = np.array([
            [0, 0, 0],
        ])
        pos2 = np.array([
            [1, 1, 1],
            [0.9, 0, 0],
        ])

        disp_tensor = matid.geometry.get_displacement_tensor(pos1, pos2)
        expected = np.array(-pos2)
        self.assertTrue(np.allclose(disp_tensor, expected))

        # Fully periodic
        disp_tensor = matid.geometry.get_displacement_tensor(pos1, pos2, pbc=True, cell=cell, mic=True)
        expected = np.array([[
            [0, 0, 0],
            [0.1, 0, 0],
        ]])
        self.assertTrue(np.allclose(disp_tensor, expected))

        # Fully periodic, reversed direction
        disp_tensor = matid.geometry.get_displacement_tensor(pos2, pos1, pbc=True, cell=cell, mic=True)
        expected = np.array([[
            [0, 0, 0],
        ], [
            [-0.1, 0, 0],
        ]])
        self.assertTrue(np.allclose(disp_tensor, expected))

        # Periodic in one direction
        disp_tensor = matid.geometry.get_displacement_tensor(pos1, pos2, pbc=[True, False, False], cell=cell, mic=True)
        expected = np.array([[
            [0, -1, -1],
            [0.1, 0, 0],
        ]])
        self.assertTrue(np.allclose(disp_tensor, expected))

    def test_to_cartesian(self):
        # Inside, unwrapped
        cell = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1]
        ])
        rel_pos = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0.5, 0.5, 0.5],
        ])
        expected_pos = np.array([
            [0, 0, 0],
            [2, 3, 1],
            [1, 1.5, 0.5],
        ])
        cart_pos = matid.geometry.to_cartesian(cell, rel_pos)
        self.assertTrue(np.allclose(cart_pos, expected_pos))

        # Outside, unwrapped
        cell = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1]
        ])
        rel_pos = np.array([
            [0, 0, 0],
            [2, 2, 2],
            [0.5, 1.5, 0.5],
        ])
        expected_pos = np.array([
            [0, 0, 0],
            [4, 6, 2],
            [1, 3.5, 0.5],
        ])
        cart_pos = matid.geometry.to_cartesian(cell, rel_pos)
        self.assertTrue(np.allclose(cart_pos, expected_pos))

        # Outside, wrapped
        cell = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1]
        ])
        rel_pos = np.array([
            [0, 0, 0],
            [2, 2, 2],
            [0.5, 1.5, 0.5],
        ])
        expected_pos = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1.5, 0.5],
        ])
        cart_pos = matid.geometry.to_cartesian(cell, rel_pos, wrap=True, pbc=True)
        self.assertTrue(np.allclose(cart_pos, expected_pos))

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GeometryTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
