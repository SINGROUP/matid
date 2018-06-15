import os
import unittest
import signal

import numpy as np
from numpy.random import RandomState

from ase import Atoms
import ase.lattice.cubic
import ase.spacegroup
from ase.visualize import view

from matid import SymmetryAnalyzer
from matid.data.constants import WYCKOFF_LETTER_POSITIONS
from matid.utils.segfault_protect import segfault_protect
import matid.geometry


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


class TestLatticeFit(unittest.TestCase):
    """Tests that the lattice fit functionality works as intended.
    """
    # def test_nacl_cubic_tilt(self):
        # # Creating a tilted NaCl conventional cell
        # cell = [
            # [5.64, 0.0, 0],
            # [0, 5.64, 0],
            # [0, 0, 5.64],
        # ]
        # nacl = Atoms(
            # cell=cell,
            # scaled_positions=[
                # [0, 0, 0],
                # [0.5, 0.5, 0],
                # [0.5, 0, 0.5],
                # [0, 0.5, 0.5],
                # [0.5, 0.5, 0.5],
                # [0.5, 0, 0],
                # [0, 0, 0.5],
                # [0, 0.5, 0],
            # ],
            # symbols=["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
            # pbc=True
        # )

        # # Get the lattice fit
        # analyzer = SymmetryAnalyzer(nacl)
        # conv_sys = analyzer.get_conventional_system()
        # conv_cell = conv_sys.get_cell()
        # print(cell)
        # print(conv_cell)
        # spg = analyzer.get_space_group_number()
        # lattice_fit = analyzer.get_conventional_lattice_fit()

        # self.assertEqual(spg, 225)
        # self.assertTrue(np.allclose(lattice_fit, cell, rtol=0, atol=1e-3))

    # def test_nacl_primitive_tilt(self):
        # # Creating a tilted NaCl conventional cell
        # cell = np.array(
            # [
                # [0.1, 2.8201, 2.8201],
                # [2.8201, 0, 2.8201],
                # [2.8201, 2.8201, 0]
            # ]
        # )
        # nacl = Atoms(
            # symbols=["Na", "Cl"],
            # scaled_positions=np.array([
                # [0, 0, 0],
                # [0.5, 0.5, 0.5]
            # ]),
            # cell=cell,
            # pbc=True
        # )

        # # Get the lattice fit
        # analyzer = SymmetryAnalyzer(nacl)
        # spg = analyzer.get_space_group_number()
        # lattice_fit = analyzer.get_conventional_lattice_fit()

        # self.assertEqual(spg, 225)

        # # This is the correct transform from primitive to conventional
        # transform = [
            # [-1, 1, 1],
            # [1, -1, 1],
            # [1, 1, -1],
        # ]
        # transformed_cell = np.dot(transform, cell)
        # self.assertTrue(np.allclose(lattice_fit, transformed_cell, rtol=0, atol=1e-3))

    # def test_supercell(self):
        # # Creating a tilted NaCl conventional cell
        # cell = [
            # [5.64, 0.5, 0],
            # [0, 5.64, 0],
            # [0, 0, 5.64],
        # ]
        # nacl = Atoms(
            # cell=cell,
            # scaled_positions=[
                # [0, 0, 0],
                # [0.5, 0.5, 0],
                # [0.5, 0, 0.5],
                # [0, 0.5, 0.5],
                # [0.5, 0.5, 0.5],
                # [0.5, 0, 0],
                # [0, 0, 0.5],
                # [0, 0.5, 0],
            # ],
            # symbols=["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
            # pbc=True
        # )
        # mult = 5
        # nacl = nacl.repeat(3*[mult])
        # orig_cell = nacl.get_cell()
        # # view(nacl)

        # # Get the conventional lattice
        # analyzer = SymmetryAnalyzer(nacl)

        # # Get the lattice fit
        # spg = analyzer.get_space_group_number()
        # lattice_fit = analyzer.get_conventional_lattice_fit()

        # self.assertEqual(spg, 225)

        # # This is the correct transform from primitive to conventional
        # transform = [
            # [0.2, 0, 0],
            # [0, 0.2, 0],
            # [0, 0, 0.2],
        # ]
        # transformed_cell = np.dot(transform, orig_cell)
        # self.assertTrue(np.allclose(lattice_fit, transformed_cell, rtol=0, atol=1e-3))

    def test_2d_primitive(self):
        orig_cell = [
            [5.64, 0.5, 0],
            [0, 5.64, 0],
            [0, 0, 5.64],
        ]
        system = Atoms(
            cell=orig_cell,
            scaled_positions=[
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0, 0],
                [0, 0, 0.5],
                [0, 0.5, 0],
            ],
            symbols=["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
            pbc=True
        )
        system = system.repeat([5, 5, 5])
        orig_cell = system.get_cell()

        # orig_cell = np.array(
            # [
                # [0.1, 2.8201, 2.8201],
                # [2.8201, 0.1, 2.8201],
                # [2.8201, 2.8201, 0]
            # ]
        # )
        # system = Atoms(
            # symbols=["Na", "Cl"],
            # scaled_positions=np.array([
                # [0, 0, 0],
                # [0.5, 0.5, 0.5]
            # ]),
            # cell=orig_cell,
            # pbc=True
        # )
        # view(system)

        # Get the conventional lattice
        analyzer = SymmetryAnalyzer(system)
        spg = analyzer.get_space_group_number()
        print(spg)
        conv_sys = analyzer.get_conventional_system()
        conv_cell = conv_sys.get_cell()
        trans = analyzer._get_spglib_transformation_matrix()
        shift = analyzer._get_spglib_origin_shift()

        # Round the coefficients to a fractional number with maximum
        # denominator of 3.
        from fractions import Fraction
        for i in range(0, trans.shape[0]):
            for j in range(0, trans.shape[0]):
                old_value = trans[i, j]
                new_value = Fraction(old_value).limit_denominator(3)
                trans[i, j] = new_value

        # Make cartesian axis swap and invert matrix
        signs = np.sign(trans)
        ones = np.array(trans)
        ones[ones != 0] = 1
        flip = np.multiply(signs, ones)

        # Invert the transformation
        trans = np.linalg.inv(trans)

        print("Shift")
        print(shift)
        print("Transf")
        print(trans)

        # Remake the conventional basis vectors in the new scaled coordinates
        new_std_lattice = np.dot(trans, orig_cell)
        new_std_lattice = np.dot(new_std_lattice, flip)
        print(new_std_lattice)
        print(conv_cell)

        # a = np.array([1, 0, 0])
        # b = np.array([0, 1, 0])
        # c = np.array([0, 0, 1])
        # print(a)
        # a_prim_scaled = np.dot(trans, a) + shift
        # b_prim_scaled = np.dot(trans, b) + shift
        # c_prim_scaled = np.dot(trans, c) + shift
        # print(a_prim_scaled)
        # print(b_prim_scaled)
        # print(c_prim_scaled)
        # a_prim = np.dot(a_prim_scaled, conv_cell.T)
        # b_prim = np.dot(b_prim_scaled, conv_cell.T)
        # c_prim = np.dot(c_prim_scaled, conv_cell.T)

        # print(a_prim)
        # print(b_prim)
        # print(c_prim)

        # lattice_fit = np.vstack((a_prim, b_prim, c_prim))
        # lattice_fit = -lattice_fit.T
        # print(lattice_fit)

        # view(conv_sys)
        # print(orig_cell)

        # # Get the lattice fit
        # spg = analyzer.get_space_group_number()
        # lattice_fit = analyzer.get_conventional_lattice_fit()

        # self.assertEqual(spg, 191)

        # # This is the correct transform from primitive to conventional
        # transform = [
            # [1, 0, 0],
            # [0, 1, 0],
            # [0, 0, 1],
        # ]
        # transformed_cell = np.dot(transform, orig_cell)
        # self.assertTrue(np.allclose(lattice_fit, transformed_cell, rtol=0, atol=1e-3))

    # def test_2d_orthogonal(self):
        # orig_cell = np.array([
            # [4.26, 0.0, 0.0],
            # [0.0, 15, 0.0],
            # [0.0, 0.0, 2.4595121467478055]
        # ])
        # system = Atoms(
            # cell=orig_cell,
            # symbols=["C", "C", "C", "C"],
            # positions=np.array((
                # [2.84, 7.5, 6.148780366869514e-1],
                # [3.55, 7.5, 1.8446341100608543],
                # [7.1e-1, 7.5, 1.8446341100608543],
                # [1.42, 7.5, 6.148780366869514e-1],
            # )),
            # pbc=[True, False, True]
        # )
        # view(system)
        # print("Original cell:")
        # print(orig_cell)
        # print(system.get_scaled_positions())

        # # Get the conventional lattice
        # analyzer = SymmetryAnalyzer(system)
        # conv_sys = analyzer.get_conventional_system()
        # conv_cell = conv_sys.get_cell()
        # print("Conventional cell:")
        # print(conv_cell)
        # view(conv_sys)

        # # Get the lattice fit
        # spg = analyzer.get_space_group_number()
        # lattice_fit = analyzer.get_conventional_lattice_fit()
        # print(lattice_fit)

        # self.assertEqual(spg, 191)

        # This is the correct transform from primitive to conventional
        # transform = [
            # [1, 0, 0],
            # [0, 1, 0],
            # [0, 0, 1],
        # ]
        # transformed_cell = np.dot(transform, orig_cell)
        # self.assertTrue(np.allclose(lattice_fit, transformed_cell, rtol=0, atol=1e-3))


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
        """Check that the volume of the lattice fit is ok.
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
        wyckoff_groups_conv = analyzer.get_wyckoff_groups_conventional()
        for group in wyckoff_groups_conv:
            if group.element == "N":
                self.assertEqual(group.wyckoff_letter, "c")
            if group.element == "B":
                self.assertEqual(group.wyckoff_letter, "a")

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
        wyckoff_groups_conv = analyzer.get_wyckoff_groups_conventional()
        for group in wyckoff_groups_conv:
            if group.element == "N":
                self.assertEqual(group.wyckoff_letter, "c")
            if group.element == "B":
                self.assertEqual(group.wyckoff_letter, "a")

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))
        # view(conv_system)


class WyckoffTests(unittest.TestCase):
    """Tests for the Wyckoff information.
    """
    def test_no_free(self):
        """Test that no Wyckoff parameter is returned when the position is not
        free.
        """
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
        """Test finding the value of the Wyckoff free parameter when two
        directions are free.
        """
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
        """Test finding the value of the Wyckoff free parameter when three
        directions are free.
        """
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


class GroundStateTests(unittest.TestCase):
    """Tests that the correct normalizer is applied to reach minimal
    configuration score that defines the Wyckoff positions.
    """
    def test_translation(self):
        """Test a tranform that translates atoms.
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

        # Check that the system has been translated correctly
        correct_translation = np.array([
            [1, 0, 0, 1/2],
            [0, 1, 0, 0],
            [0, 0, 1, 1/2],
            [0, 0, 0, 1],
        ])
        conv_system = analyzer.get_conventional_system()
        conv_pos = conv_system.get_scaled_positions()
        orig_pos = np.zeros((4, 4))
        orig_pos[:, 3] = 1
        orig_pos[:, 0:3] = system.get_scaled_positions()

        assumed_pos = np.dot(orig_pos, correct_translation.T)
        assumed_pos = assumed_pos[:, 0:3]
        assumed_pos = matid.geometry.get_wrapped_positions(assumed_pos)
        self.assertTrue(np.array_equal(assumed_pos, conv_pos))

    def test_transformation_affine(self):
        """Test a tranform where the transformation is a proper rigid
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
        # space_group = analyzer.get_space_group_number()
        # print(space_group)
        # view(system)

        # The assumed ground state
        analyzer = SymmetryAnalyzer(system)
        # conv_system = analyzer.get_conventional_system()
        # view(conv_system)

        # Check that the correct Wyckoff positions are occupied, and that an
        # axis swap transformation has not been applied.
        wyckoff_groups_conv = analyzer.get_wyckoff_groups_conventional()
        for group in wyckoff_groups_conv:
            if group.element == "H":
                self.assertEqual(group.wyckoff_letter, "a")
            if group.element == "C":
                self.assertEqual(group.wyckoff_letter, "c")


if __name__ == '__main__':
    suites = []
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(TestSegfaultProtect))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestLatticeFit))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyser3DTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyser2DTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(WyckoffTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(GroundStateTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
