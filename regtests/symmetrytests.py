from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
import signal

import numpy as np
from numpy.random import RandomState

from ase import Atoms
import ase.lattice.cubic
import ase.spacegroup
import ase.build

from matid import SymmetryAnalyzer
from matid.symmetry.wyckoffset import WyckoffSet
from matid.data.constants import WYCKOFF_LETTER_POSITIONS
from matid.utils.segfault_protect import segfault_protect


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TestSegfaultProtect(unittest.TestCase):
    """Test that the wrapper function for guarding against SIGSEGV works as
    intended.
    """
    def test_sigsegv_wo_args(self):

        def sigsegv():
            os.kill(os.getpid(), signal.SIGSEGV)

        with self.assertRaises(RuntimeError):
            segfault_protect(sigsegv)

    def test_sigsegv_w_args(self):

        def sigsegv(a):
            os.kill(os.getpid(), signal.SIGSEGV)

        with self.assertRaises(RuntimeError):
            segfault_protect(sigsegv, "moi")


class SymmetryAnalyser3DTests(unittest.TestCase):
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
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_sets_conv)

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
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_sets_conv)

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
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_sets_conv)

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
        self.assertWyckoffGroupsOk(data.conv_system, data.wyckoff_sets_conv)

    def assertWyckoffGroupsOk(self, system, wyckoff_sets):
        """Check that the Wyckoff sets contain all atoms and are ordered
        """
        prev_w_index = None
        prev_z = None
        n_atoms = len(system)
        n_atoms_wyckoff = 0
        for wset in wyckoff_sets:

            # Check that the current Wyckoff letter index is greater than
            # previous, if not the atomic number must be greater
            wyckoff_letter = wset.wyckoff_letter
            atomic_number = wset.atomic_number
            i_w_index = WYCKOFF_LETTER_POSITIONS[wyckoff_letter]
            if prev_w_index is not None:
                self.assertGreaterEqual(i_w_index, prev_w_index)
                if i_w_index == prev_w_index:
                    self.assertGreater(atomic_number, prev_z)

            prev_w_index = i_w_index
            prev_z = atomic_number

            # Gather the number of atoms in eaach set to see that it matches
            # the amount of atoms in the system
            n = len(wset.indices)
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
        data.wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
        data.prim_wyckoff = analyzer.get_wyckoff_letters_primitive()
        data.prim_equiv = analyzer.get_equivalent_atoms_primitive()
        data.equivalent_original = analyzer.get_equivalent_atoms_original()
        data.equivalent_conv = analyzer.get_equivalent_atoms_conventional()
        data.has_free_wyckoff_parameters = analyzer.get_has_free_wyckoff_parameters()
        data.chiral = analyzer.get_is_chiral()

        return data


class SymmetryAnalyser2DTests(unittest.TestCase):
    """Tests the analysis of bulk 2D materials.
    """
    def test_graphene_primitive(self):
        # Original system in positions: C: d
        system = Atoms(
            symbols=["C", "C"],
            cell=np.array((
                [2.4595121467478055, 0.0, 0.0],
                [-1.2297560733739028, 2.13, 0.0],
                [0.0, 0.0, 20.0]
            )),
            scaled_positions=np.array((
                [1/3, 2/3, 0.5],
                [2/3, 1/3, 0.5]
            )),
            pbc=[True, True, False]
        )
        # view(system)

        analyzer = SymmetryAnalyzer(system)
        wyckoff_letters_conv = analyzer.get_wyckoff_letters_conventional()
        wyckoff_letters_assumed = ["c", "c"]
        self.assertTrue(np.array_equal(wyckoff_letters_assumed, wyckoff_letters_conv))

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))
        # view(conv_system)

    def test_graphene_primitive_basis_swap(self):
        """Tests a system where the cartesian coordinates will get rotated in
        the conventional system.
        """
        # Original system in positions: C: d
        system = Atoms(
            symbols=["C", "C"],
            cell=np.array((
                [2.4595121467478055, 0.0, 0.0],
                [-1.2297560733739028, 0.0, 2.13],
                [0.0, 20.0, 0.0],
            )),
            scaled_positions=np.array((
                [1/3, 2/3, 0.5],
                [2/3, 1/3, 0.5]
            )),
            pbc=[True, True, False]
        )
        # view(system)

        analyzer = SymmetryAnalyzer(system)
        wyckoff_letters_conv = analyzer.get_wyckoff_letters_conventional()
        wyckoff_letters_assumed = ["c", "c"]
        self.assertTrue(np.array_equal(wyckoff_letters_assumed, wyckoff_letters_conv))

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))
        # view(conv_system)

    def test_boron_nitride_primitive(self):
        """Test a system where the normalized system cannot be correctly found
        if the flatness of the structure is not taken into account. Due to the
        structure being flat, any non-rigid transformations in the flat
        directions do not affect the rigidity, and are thus allowed.
        """
        # Original system in positions: B: a, N: e. This can only be converted
        # to the ground state: B: a, N: c if an inversion is performed.
        system = Atoms(
            symbols=["B", "N"],
            cell=np.array((
                [2.4595121467478055, 0.0, 0.0],
                [-1.2297560733739028, 2.13, 0.0],
                [0.0, 0.0, 20.0]
            )),
            scaled_positions=np.array((
                [0, 0, 0.0],
                [2/3, 1/3, 0.0]
            )),
            pbc=[True, True, False]
        )
        # view(system)

        analyzer = SymmetryAnalyzer(system)

        # Check that the correct Wyckoff positions are present in the
        # normalized system
        wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
        for wset in wyckoff_sets_conv:
            if wset.element == "N":
                self.assertEqual(wset.wyckoff_letter, "c")
            if wset.element == "B":
                self.assertEqual(wset.wyckoff_letter, "a")

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))
        # view(conv_system)

    def test_boron_nitride_rotated(self):
        """Test a system where the non-periodic direction is not the third axis
        direction in the original system, but in the normalized system the
        third axis will be the non-periodic one.
        """
        # Original system in positions: B: a, N: e. This can only be converted
        # to the ground state: B: a, N: c if an inversion is performed.
        system = Atoms(
            symbols=["B", "N"],
            cell=np.array((
                [2.4595121467478055, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [-1.2297560733739028, 0.0, 2.13],
            )),
            scaled_positions=np.array((
                [0, 0, 0.0],
                [2/3, 0.0, 1/3]
            )),
            pbc=[True, False, True]
        )
        # view(system)

        analyzer = SymmetryAnalyzer(system)

        # Check that the correct Wyckoff positions are present in the
        # normalized system
        wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
        for wset in wyckoff_sets_conv:
            if wset.element == "N":
                self.assertEqual(wset.wyckoff_letter, "c")
            if wset.element == "B":
                self.assertEqual(wset.wyckoff_letter, "a")

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))
        # view(conv_system)


class WyckoffTests(unittest.TestCase):
    """Tests for the Wyckoff information.
    """
    # def test_non_default_68(self):
        # """Tests that systems which deviate from the default settings (spglib
        # does not use the default settings, but instead will use the setting
        # with lowest Hall number) are handled correctly.
        # """
        # sg_68 = Atoms(
            # symbols=["Au", "Au", "Sn", "Sn", "Sn", "Sn", "Sn", "Sn", "Sn", "Sn"],
            # scaled_positions=[
                # [0.9852540807, 0.0000000000, 0.9926270404],
                # [0.9852540807, 0.5000004882, 0.4926265201],
                # [0.7215549731, 0.3327377765, 0.0235395469],
                # [0.7215549731, 0.8327367459, 0.1980144280],
                # [0.2456177163, 0.6668151491, 0.2855491427],
                # [0.2456177163, 0.1668146609, 0.4600695715],
                # [0.7215549731, 0.1672627118, 0.5235405450],
                # [0.7215549731, 0.6672632000, 0.6980154261],
                # [0.2456177163, 0.8331847967, 0.7855486225],
                # [0.2456177163, 0.3331858273, 0.9600690513],
            # ],
            # cell= [
                # [0.000000, -3.293253, 5.939270],
                # [6.584074, 0.000000, 0.000000],
                # [0.000000, 6.586507, 0.000000],
            # ],
            # pbc=True
        # )

        # # Find the Wyckoff groups
        # analyzer = SymmetryAnalyzer(sg_68)
        # spg = analyzer.get_space_group_number()
        # self.assertEqual(spg, 68)
        # norm_sys = analyzer.get_conventional_system()
        # # from ase.visualize import view
        # # view(norm_sys)
        
        # # wyckoff_letters = analyzer.get_wyckoff_letters_conventional()
        # # print(wyckoff_letters)
        # wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # # Check that groups are correct
        # expected_sets = [
            # WyckoffSet("a", 79, "Au", space_group=68, multiplicity=4),
            # WyckoffSet("i", 50, "Sn", x=0.16276206030000004, y=0.11815044619999993, z=-0.4172622234999999, space_group=68, multiplicity=16),
        # ]

        # for w1, w2 in zip(expected_sets, wyckoff_sets):
            # self.assertEqual(w1, w2)

    # def test_non_default_129(self):
        # """Tests that systems which deviate from the default settings (spglib
        # does not use the default settings, but instead will use the setting
        # with lowest Hall number) are handled correctly.
        # """
        # sg_129 = Atoms(
            # symbols=["F", "F", "Nd", "Nd", "S", "S"],
            # scaled_positions=[
                # [0.7499993406, 0.2499997802, 0.0000000000],
                # [0.2499997802, 0.7499993406, 0.0000000000],
                # [0.2499997802, 0.2499997802, 0.2301982694],
                # [0.7499993406, 0.7499993406, 0.7698006940],
                # [0.7499993406, 0.7499993406, 0.3524397980],
                # [0.2499997802, 0.2499997802, 0.6475606156],
            # ],
            # cell= [
                # [3.919363, 0.000000, 0.000000],
                # [0.000000, 3.919363, 0.000000],
                # [0.000000, 0.000000, 6.895447],
            # ],
            # pbc=True
        # )

        # # Find the Wyckoff groups
        # analyzer = SymmetryAnalyzer(sg_129)
        # wyckoff_sets = analyzer.get_wyckoff_sets_conventional()
        # spg = analyzer.get_space_group_number()
        # self.assertEqual(spg, 129)

        # # Check that groups are correct
        # expected_sets = [
            # WyckoffSet("a", 9, "F", space_group=129, multiplicity=2),
            # WyckoffSet("c", 16, "S", z=0.352439798, space_group=129, multiplicity=2),
            # WyckoffSet("c", 60, "Nd", z=0.7698017306, space_group=129, multiplicity=2),
        # ]
        # for w1, w2 in zip(expected_sets, wyckoff_sets):
            # self.assertEqual(w1, w2)

    # def test_default_225(self):
        # """Tests that systems which deviate from the default settings (spglib
        # does not use the default settings, but instead will use the setting
        # with lowest Hall number) are handled correctly.
        # """
        # # Create structure that has space group 129: the origin setting differ
        # # from default settings.
        # a = 2.87
        # system = ase.spacegroup.crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # # # view(fcc)

        # # Find the Wyckoff groups
        # analyzer = SymmetryAnalyzer(system)
        # wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # # Check that groups are correct
        # expected_sets = [
            # WyckoffSet("a", 13, "Al", space_group=225, multiplicity=4),
        # ]
        # for w1, w2 in zip(expected_sets, wyckoff_sets):
            # self.assertEqual(w1, w2)

    # def test_no_free(self):
        # """Test that no Wyckoff parameter is returned when the position is not
        # free.
        # """
        # # Create structure
        # a = 2.87
        # fcc = ase.spacegroup.crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # # view(fcc)

        # # Find the Wyckoff groups
        # analyzer = SymmetryAnalyzer(fcc)
        # wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # # Check that the information matches
        # self.assertEqual(len(wyckoff_sets), 1)
        # wset = wyckoff_sets[0]
        # self.assertEqual(wset.atomic_number, 13)
        # self.assertEqual(wset.element, "Al")
        # self.assertEqual(wset.wyckoff_letter, "a")
        # self.assertEqual(wset.indices, [0, 1, 2, 3])
        # self.assertEqual(wset.x, None)
        # self.assertEqual(wset.y, None)
        # self.assertEqual(wset.z, None)

    def test_one_free(self):
        """Test finding the value of the Wyckoff free parameter when one
        direction is free.
        """
        # Create structure
        free_variables = {
            "x": 0.13
        }
        a = 12
        fcc = ase.spacegroup.crystal('Al', [(0, free_variables["x"], 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # view(fcc)

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        norm_sys = analyzer.get_conventional_system()
        from ase.visualize import view
        view(norm_sys)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that the information matches
        self.assertEqual(len(wyckoff_sets), 1)
        wset = wyckoff_sets[0]
        self.assertEqual(wset.atomic_number, 13)
        self.assertEqual(wset.element, "Al")
        self.assertEqual(wset.wyckoff_letter, "e")
        self.assertEqual(wset.indices, list(range(len(fcc))))
        for var, value in free_variables.items():
            calculated_value = getattr(wset, var)
            self.assertTrue(calculated_value - value <= 1e-2)

    # def test_two_free(self):
        # """Test finding the value of the Wyckoff free parameter when two
        # directions are free.
        # """
        # # Create structure
        # free_variables = {
            # "y": 0.13,
            # "z": 0.077,
        # }
        # a = 40
        # fcc = ase.spacegroup.crystal('Al', [(0, free_variables["y"], free_variables["z"])], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # # view(fcc)

        # # Find the Wyckoff groups
        # analyzer = SymmetryAnalyzer(fcc)
        # wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # # Check that the information matches
        # self.assertEqual(len(wyckoff_sets), 1)
        # wset = wyckoff_sets[0]
        # self.assertEqual(wset.atomic_number, 13)
        # self.assertEqual(wset.element, "Al")
        # self.assertEqual(wset.wyckoff_letter, "j")
        # self.assertEqual(wset.indices, list(range(len(fcc))))
        # for var, value in free_variables.items():
            # calculated_value = getattr(wset, var)
            # self.assertTrue(calculated_value - value <= 1e-2)

    # def test_three_free(self):
        # """Test finding the value of the Wyckoff free parameter when three
        # directions are free.
        # """
        # # Create structure
        # free_variables = {
            # "x": 0.36,
            # "y": 0.13,
            # "z": 0.077,
        # }
        # a = 100
        # fcc = ase.spacegroup.crystal('Al', [(free_variables["x"], free_variables["y"], free_variables["z"])], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        # # view(fcc)

        # # Find the Wyckoff groups
        # analyzer = SymmetryAnalyzer(fcc)
        # wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # # Check that the information matches
        # self.assertEqual(len(wyckoff_sets), 1)
        # wset = wyckoff_sets[0]
        # self.assertEqual(wset.atomic_number, 13)
        # self.assertEqual(wset.element, "Al")
        # self.assertEqual(wset.wyckoff_letter, "l")
        # self.assertEqual(wset.indices, list(range(len(fcc))))
        # for var, value in free_variables.items():
            # calculated_value = getattr(wset, var)
            # self.assertTrue(calculated_value - value <= 1e-2)


class GroundStateTests(unittest.TestCase):
    """Tests that the correct normalizer is applied to reach minimal
    configuration score that defines the Wyckoff positions.
    """
    def test_translation(self):
        """Test a transform that translates atoms.
        """
        # The original system belongs to space group 12, see
        # http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-normsets?from=wycksets&gnum=12
        # and
        # http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-wp-list?gnum=012
        system = Atoms(
            cell=[
                [3.3, 0., 0.],
                [0., 1., 0.],
                [-1., 0., 3.],
            ],
            scaled_positions=[
                [0.5, 0.5, 0.],
                [0.5, 0.,  0.5],
                [0.,  0.,  0.],
                [0.,  0.5, 0.5],
            ],
            symbols=["C", "H", "C", "H"],
            pbc=True
        )
        # The assumed ground state
        correct_state = ["d", "a", "d", "a"]
        analyzer = SymmetryAnalyzer(system)
        orig_wyckoffs = analyzer.get_wyckoff_letters_original()
        self.assertTrue(np.array_equal(orig_wyckoffs, correct_state))

        # Check that the system has been translated correctly. The correct
        # positions can be seen at
        # http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-wp-list?gnum=012
        conv_system = analyzer.get_conventional_system()
        conv_pos = conv_system.get_scaled_positions()

        a1 = [0.0, 0.0, 0.0]
        a2 = [0.5, 0.5, 0.0]
        d1 = [0, 0.5, 0.5]
        d2 = [0.5, 0.0, 0.5]

        # Test that the Wyckoff positions d are correct, order does not matter
        pos1 = np.array_equal(conv_pos[0], d1)
        if pos1:
            self.assertTrue(np.array_equal(conv_pos[2], d2))
        else:
            self.assertTrue(np.array_equal(conv_pos[0], d2))
            self.assertTrue(np.array_equal(conv_pos[2], d1))

        # Test that the Wyckoff positions a are correct, order does not matter
        pos1 = np.array_equal(conv_pos[1], a1)
        if pos1:
            self.assertTrue(np.array_equal(conv_pos[3], a2))
        else:
            self.assertTrue(np.array_equal(conv_pos[1], a2))
            self.assertTrue(np.array_equal(conv_pos[3], a1))

    def test_transformation_affine(self):
        """Test a transform where the transformation is a proper rigid
        transformation in the scaled cell basis, but will be non-rigid in the
        cartesian basis. This kind of transformations should not be allowed.
        """
        system = Atoms(
            cell=[
                [1, 0., 0.],
                [0., 2.66, 0.],
                [0., 0., 1.66],
            ],
            scaled_positions=[
                [0.0, 1/2, 0.0],
                [0.0, 0.0, 0.0],
            ],
            symbols=["H", "C"],
            pbc=True
        )
        analyzer = SymmetryAnalyzer(system)
        space_group = analyzer.get_space_group_number()

        # The assumed ground state
        analyzer = SymmetryAnalyzer(system)
        conv_system = analyzer.get_conventional_system()

        # Check that the correct Wyckoff positions are occupied, and that an
        # axis swap transformation has not been applied.
        wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
        for wset in wyckoff_sets_conv:
            if wset.element == "H":
                self.assertEqual(wset.wyckoff_letter, "a")
            if wset.element == "C":
                self.assertEqual(wset.wyckoff_letter, "c")

    # def test_zinc_blende(self):
        # """Tests that all different forms of the zinc-blende structure can be
        # normalized to the same structure. The use of improper normalizers
        # from Bilbao is required to achieve this, but the resulting structure
        # must then have a rigid translation in cartesian coordinates to the
        # original system.
        # """
        # # Primitive
        # zb_prim = ase.build.bulk("ZnS", crystalstructure="zincblende", a=5.42)
        # analyzer_prim = SymmetryAnalyzer(zb_prim)
        # zb_prim_conv = analyzer_prim.get_conventional_system()
        # wyckoff_prim = analyzer_prim.get_wyckoff_letters_conventional()
        # pos_prim = zb_prim_conv.get_positions()
        # z_prim = zb_prim_conv.get_atomic_numbers()

        # # Conventional
        # zb_conv = ase.build.bulk("ZnS", crystalstructure="zincblende", a=5.42, cubic=True)
        # analyzer_conv = SymmetryAnalyzer(zb_conv)
        # zb_conv_conv = analyzer_conv.get_conventional_system()
        # wyckoff_conv = analyzer_conv.get_wyckoff_letters_conventional()
        # pos_conv = zb_conv_conv.get_positions()
        # z_conv = zb_conv_conv.get_atomic_numbers()
        # # print(wyckoff_conv)
        # # print(pos_conv)

        # # Rotate along x -90
        # # transform = np.array([
            # # [1, 0, 0],
            # # [0, 0, 1],
            # # [0, -1, 0],
        # # ])
        # # pos_conv = np.dot(pos_conv, transform)
        # # zb_conv_conv.set_positions(pos_conv)

        # # Inversion
        # # transform = np.array([
            # # [-1, 0, 0],
            # # [0, -1, 0],
            # # [0, 0, -1],
        # # ])
        # # pos_conv = zb_conv_conv.get_scaled_positions()
        # # pos_conv = np.dot(pos_conv, transform)
        # # zb_conv_conv.set_scaled_positions(pos_conv)

        # # zb_conv_conv.set_pbc(True)
        # # zb_conv_conv.wrap()
        # # pos_conv = zb_conv_conv.get_positions()

        # # Assume that the same atoms are at same positions
        # for ipos, iz, iw in zip(pos_prim, z_prim, wyckoff_prim):
            # found = False
            # for jpos, jz, jw in zip(pos_prim, z_prim, wyckoff_prim):
                # match_pos = np.allclose(ipos, jpos)
                # match_z = iz == jz
                # match_w = iw == jw
                # if match_pos and match_z and match_w:
                    # found = True
            # self.assertTrue(found)

        # from ase.visualize import view
        # view(zb_prim_conv)
        # view(zb_conv_conv)


if __name__ == '__main__':
    suites = []
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(TestSegfaultProtect))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyser3DTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyser2DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(WyckoffTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(GroundStateTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
