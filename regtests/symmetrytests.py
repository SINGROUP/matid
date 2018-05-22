import unittest
import numpy as np
from ase import Atoms
import ase.lattice.cubic
import ase.spacegroup
from ase.visualize import view
from systax import SymmetryAnalyzer
from systax.data.constants import WYCKOFF_LETTER_POSITIONS
from numpy.random import RandomState


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SymmetryAnalyserTests(unittest.TestCase):
    """Tests the analysis of bulk 3D materials.
    """
    def test_diamond(self):
        """Test that a silicon diamond lattice is characterized correctly.
        """
        # Create the system
        si = ase.lattice.cubic.Diamond(
            size=(1, 1, 1),
            symbol='Si',
            pbc=(1, 1, 1),
            latticeconstant=5.430710)

        # Apply some noise
        si.rattle(stdev=0.05, seed=42)
        si.translate([1, 2, 1])
        cell = si.get_cell()
        a = cell[0, :]
        a *= 1.02
        cell[0, :] = a
        si.set_cell(cell)

        # Get the data
        data = self.get_material3d_properties(si)

        # Check that the data is valid
        self.assertEqual(data.chiral, False)
        self.assertEqual(data.space_group_number, 227)
        self.assertEqual(data.space_group_int, "Fd-3m")
        self.assertEqual(data.hall_symbol, "F 4d 2 3 -1d")
        self.assertEqual(data.hall_number, 525)
        self.assertEqual(data.point_group, "m-3m")
        self.assertEqual(data.crystal_system, "cubic")
        self.assertEqual(data.bravais_lattice, "cF")
        self.assertEqual(data.choice, "1")
        self.assertTrue(np.array_equal(data.equivalent_conv, [0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(data.wyckoff_conv, ["a", "a", "a", "a", "a", "a", "a", "a"]))
        self.assertTrue(np.array_equal(data.equivalent_original, [0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(data.wyckoff_original, ["a", "a", "a", "a", "a", "a", "a", "a"]))
        self.assertTrue(np.array_equal(data.prim_wyckoff, ["a", "a"]))
        self.assertTrue(np.array_equal(data.prim_equiv, [0, 0]))
        self.assertFalse(data.has_free_wyckoff_parameters)
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_groups_conv)
        self.assertVolumeOk(si, data.conv_system, data.lattice_fit)

    def test_fcc(self):
        """Test that a primitive NaCl fcc lattice is characterized correctly.
        """
        # Create the system
        cell = np.array(
            [
                [0, 2.8201, 2.8201],
                [2.8201, 0, 2.8201],
                [2.8201, 2.8201, 0]
            ]
        )
        cell[0, :] *= 1.05
        nacl = Atoms(
            symbols=["Na", "Cl"],
            scaled_positions=np.array([
                [0, 0, 0],
                [0.5, 0.5, 0.5]
            ]),
            cell=cell,
            pbc=True
        )
        nacl = nacl.repeat([2, 1, 1])

        # Get the data
        data = self.get_material3d_properties(nacl)

        # Check that the data is valid
        self.assertEqual(data.space_group_number, 225)
        self.assertEqual(data.space_group_int, "Fm-3m")
        self.assertEqual(data.hall_symbol, "-F 4 2 3")
        self.assertEqual(data.hall_number, 523)
        self.assertEqual(data.point_group, "m-3m")
        self.assertEqual(data.crystal_system, "cubic")
        self.assertEqual(data.bravais_lattice, "cF")
        self.assertEqual(data.choice, "")
        self.assertTrue(np.array_equal(data.equivalent_conv, [0, 1, 0, 1, 0, 1, 0, 1]))
        self.assertTrue(np.array_equal(data.wyckoff_conv, ["a", "b", "a", "b", "a", "b", "a", "b"]))
        self.assertTrue(np.array_equal(data.equivalent_original, [0, 1, 0, 1]))
        self.assertTrue(np.array_equal(data.wyckoff_original, ["a", "b", "a", "b"]))
        self.assertTrue(np.array_equal(data.prim_equiv, [0, 1]))
        self.assertTrue(np.array_equal(data.prim_wyckoff, ["a", "b"]))
        self.assertFalse(data.has_free_wyckoff_parameters)
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_groups_conv)
        self.assertVolumeOk(nacl, data.conv_system, data.lattice_fit)

    def test_bcc(self):
        """Test that a body centered cubic lattice for copper is characterized
        correctly.
        """
        from ase.lattice.cubic import BodyCenteredCubic
        system = BodyCenteredCubic(
            directions=[[1, 0, 0], [0, 1, 0], [1, 1, 1]],
            size=(1, 1, 1),
            symbol='Cu',
            pbc=True,
            latticeconstant=4.0)

        # Get the data
        data = self.get_material3d_properties(system)

        # Check that the data is valid
        self.assertEqual(data.space_group_number, 229)
        self.assertEqual(data.space_group_int, "Im-3m")
        self.assertEqual(data.hall_symbol, "-I 4 2 3")
        self.assertEqual(data.hall_number, 529)
        self.assertEqual(data.point_group, "m-3m")
        self.assertEqual(data.crystal_system, "cubic")
        self.assertEqual(data.bravais_lattice, "cI")
        self.assertEqual(data.choice, "")
        self.assertTrue(np.array_equal(data.equivalent_conv, [0, 0]))
        self.assertTrue(np.array_equal(data.wyckoff_conv, ["a", "a"]))
        self.assertTrue(np.array_equal(data.equivalent_original, [0]))
        self.assertTrue(np.array_equal(data.wyckoff_original, ["a"]))
        self.assertTrue(np.array_equal(data.prim_equiv, [0]))
        self.assertTrue(np.array_equal(data.prim_wyckoff, ["a"]))
        self.assertFalse(data.has_free_wyckoff_parameters)
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_groups_conv)
        self.assertVolumeOk(system, data.conv_system, data.lattice_fit)

    def test_unsymmetric(self):
        """Test that a random system is handled correctly.
        """
        rng = RandomState(42)
        positions = 10*rng.rand(10, 3)
        system = Atoms(
            positions=positions,
            symbols=["H", "C", "Na", "Fe", "Cu", "He", "Ne", "Mg", "Si", "Ti"],
            cell=[10, 10, 10],
            pbc=True
        )

        # Get the data
        data = self.get_material3d_properties(system)

        # Check that the data is valid
        self.assertEqual(data.space_group_number, 1)
        self.assertEqual(data.space_group_int, "P1")
        self.assertEqual(data.hall_number, 1)
        self.assertEqual(data.point_group, "1")
        self.assertEqual(data.crystal_system, "triclinic")
        self.assertEqual(data.bravais_lattice, "aP")
        self.assertTrue(data.has_free_wyckoff_parameters)
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_groups_conv)
        self.assertVolumeOk(system, data.conv_system, data.lattice_fit)

    def assertVolumeOk(self, orig_sys, conv_sys, lattice_fit):
        """Check that the Wyckoff groups contain all atoms and are ordered
        """
        n_atoms_orig = len(orig_sys)
        volume_orig = orig_sys.get_volume()
        n_atoms_conv = len(conv_sys)
        volume_conv = np.linalg.det(lattice_fit)
        self.assertTrue(np.allclose(volume_orig/n_atoms_orig, volume_conv/n_atoms_conv, atol=1e-8))

    def assertWyckoffGroupsOk(self, system, wyckoff_groups):
        """Check that the Wyckoff groups contain all atoms and are ordered
        """
        prev_w_index = None
        prev_z = None
        n_atoms = len(system)
        n_atoms_wyckoff = 0
        for group in wyckoff_groups:

            # Check that the current Wyckoff letter index is greater than
            # previous, if not the atomic number must be greater
            wyckoff_letter = group.wyckoff_letter
            atomic_number = group.atomic_number
            i_w_index = WYCKOFF_LETTER_POSITIONS[wyckoff_letter]
            if prev_w_index is not None:
                self.assertGreaterEqual(i_w_index, prev_w_index)
                if i_w_index == prev_w_index:
                    self.assertGreater(atomic_number, prev_z)

            prev_w_index = i_w_index
            prev_z = atomic_number

            # Gather the number of atoms in eaach group to see that it matches
            # the amount of atoms in the system
            n = len(group.indices)
            n_atoms_wyckoff += n

        self.assertEqual(n_atoms, n_atoms_wyckoff)

    def get_material3d_properties(self, system):
        analyzer = SymmetryAnalyzer(system)
        data = dotdict()

        data.space_group_number = analyzer.get_space_group_number()
        data.space_group_int = analyzer.get_space_group_international_short()
        data.hall_symbol = analyzer.get_hall_symbol()
        data.hall_number = analyzer.get_hall_number()
        data.conv_system = analyzer.get_conventional_system()
        data.prim_system = analyzer.get_primitive_system()
        data.translations = analyzer.get_translations()
        data.rotations = analyzer.get_rotations()
        data.origin_shift = analyzer._get_spglib_origin_shift()
        data.choice = analyzer.get_choice()
        data.point_group = analyzer.get_point_group()
        data.crystal_system = analyzer.get_crystal_system()
        data.bravais_lattice = analyzer.get_bravais_lattice()
        data.transformation_matrix = analyzer._get_spglib_transformation_matrix()
        data.wyckoff_original = analyzer.get_wyckoff_letters_original()
        data.wyckoff_conv = analyzer.get_wyckoff_letters_conventional()
        data.wyckoff_groups_conv = analyzer.get_wyckoff_groups_conventional()
        data.prim_wyckoff = analyzer.get_wyckoff_letters_primitive()
        data.prim_equiv = analyzer.get_equivalent_atoms_primitive()
        data.equivalent_original = analyzer.get_equivalent_atoms_original()
        data.equivalent_conv = analyzer.get_equivalent_atoms_conventional()
        data.lattice_fit = analyzer.get_conventional_lattice_fit()
        data.has_free_wyckoff_parameters = analyzer.get_has_free_wyckoff_parameters()
        data.chiral = analyzer.get_is_chiral()

        return data


class WyckoffTests(unittest.TestCase):
    """Tests for the Wyckoff information.
    """
    def test_no_free(self):
        # Create structure
        a = 2.87
        fcc = ase.spacegroup.crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # view(fcc)

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_groups = analyzer.get_wyckoff_groups_conventional()

        # Check that the information matches
        self.assertEqual(len(wyckoff_groups), 1)
        group = wyckoff_groups[0]
        self.assertEqual(group.atomic_number, 13)
        self.assertEqual(group.element, "Al")
        self.assertEqual(group.wyckoff_letter, "a")
        self.assertEqual(group.indices, [0, 1, 2, 3])
        self.assertEqual(group.x, None)
        self.assertEqual(group.y, None)
        self.assertEqual(group.z, None)

    def test_one_free(self):
        # Create structure
        free_variables = {
            "x": 0.13
        }
        a = 12
        fcc = ase.spacegroup.crystal('Al', [(0, free_variables["x"], 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # view(fcc)

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_groups = analyzer.get_wyckoff_groups_conventional()

        # Check that the information matches
        self.assertEqual(len(wyckoff_groups), 1)
        group = wyckoff_groups[0]
        self.assertEqual(group.atomic_number, 13)
        self.assertEqual(group.element, "Al")
        self.assertEqual(group.wyckoff_letter, "e")
        self.assertEqual(group.indices, list(range(len(fcc))))
        for var, value in free_variables.items():
            calculated_value = getattr(group, var)
            self.assertTrue(calculated_value - value <= 1e-2)

    def test_two_free(self):
        # Create structure
        free_variables = {
            "y": 0.13,
            "z": 0.077,
        }
        a = 40
        fcc = ase.spacegroup.crystal('Al', [(0, free_variables["y"], free_variables["z"])], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # view(fcc)

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_groups = analyzer.get_wyckoff_groups_conventional()

        # Check that the information matches
        self.assertEqual(len(wyckoff_groups), 1)
        group = wyckoff_groups[0]
        self.assertEqual(group.atomic_number, 13)
        self.assertEqual(group.element, "Al")
        self.assertEqual(group.wyckoff_letter, "j")
        self.assertEqual(group.indices, list(range(len(fcc))))
        for var, value in free_variables.items():
            calculated_value = getattr(group, var)
            self.assertTrue(calculated_value - value <= 1e-2)

    def test_three_free(self):
        # Create structure
        free_variables = {
            "x": 0.36,
            "y": 0.13,
            "z": 0.077,
        }
        a = 100
        fcc = ase.spacegroup.crystal('Al', [(free_variables["x"], free_variables["y"], free_variables["z"])], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # view(fcc)

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_groups = analyzer.get_wyckoff_groups_conventional()

        # Check that the information matches
        self.assertEqual(len(wyckoff_groups), 1)
        group = wyckoff_groups[0]
        self.assertEqual(group.atomic_number, 13)
        self.assertEqual(group.element, "Al")
        self.assertEqual(group.wyckoff_letter, "l")
        self.assertEqual(group.indices, list(range(len(fcc))))
        for var, value in free_variables.items():
            calculated_value = getattr(group, var)
            self.assertTrue(calculated_value - value <= 1e-2)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyserTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(WyckoffTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
