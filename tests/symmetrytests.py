import os
import unittest
import signal

import numpy as np
from numpy.random import RandomState

from ase import Atoms
import ase.lattice.cubic
import ase.spacegroup
import ase.build
from ase.visualize import view

from matid import SymmetryAnalyzer
from matid.symmetry.wyckoffset import WyckoffSet
from matid.data.constants import WYCKOFF_LETTER_POSITIONS
from matid.utils.segfault_protect import segfault_protect

from conftest import create_graphene, create_si, create_mos2


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
        si = create_si()

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
        pbc_conv = data.conv_system.get_pbc()
        pbc_prim = data.prim_system.get_pbc()
        self.assertTrue(np.array_equal(pbc_conv, [True, True, True]))
        self.assertTrue(np.array_equal(pbc_prim, [True, True, True]))

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
        pbc_conv = data.conv_system.get_pbc()
        pbc_prim = data.prim_system.get_pbc()
        self.assertTrue(np.array_equal(pbc_conv, [True, True, True]))
        self.assertTrue(np.array_equal(pbc_prim, [True, True, True]))

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
        pbc_conv = data.conv_system.get_pbc()
        pbc_prim = data.prim_system.get_pbc()
        self.assertTrue(np.array_equal(pbc_conv, [True, True, True]))
        self.assertTrue(np.array_equal(pbc_prim, [True, True, True]))

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
        pbc_conv = data.conv_system.get_pbc()
        pbc_prim = data.prim_system.get_pbc()
        self.assertTrue(np.array_equal(pbc_conv, [True, True, True]))
        self.assertTrue(np.array_equal(pbc_prim, [True, True, True]))

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
        system = create_graphene()
        analyzer = SymmetryAnalyzer(system)
        wyckoff_letters_conv = analyzer.get_wyckoff_letters_conventional()
        wyckoff_letters_assumed = ["c", "c"]
        self.assertTrue(np.array_equal(wyckoff_letters_assumed, wyckoff_letters_conv))

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))

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

        analyzer = SymmetryAnalyzer(system)
        wyckoff_letters_conv = analyzer.get_wyckoff_letters_conventional()
        wyckoff_letters_assumed = ["c", "c"]
        self.assertTrue(np.array_equal(wyckoff_letters_assumed, wyckoff_letters_conv))

        conv_system = analyzer.get_conventional_system()
        pbc = conv_system.get_pbc()
        self.assertTrue(np.array_equal(pbc, [True, True, False]))

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

    def test_mos2(self):
        """Tests a non-flat 2D system with vacuum.
        """
        system = create_mos2()
        analyzer = SymmetryAnalyzer(system)
        conv_system = analyzer.get_conventional_system()
        prim_system = analyzer.get_primitive_system()
        self.assertTrue(np.array_equal(conv_system.get_pbc(), [True, True, False]))
        self.assertTrue(np.array_equal(prim_system.get_pbc(), [True, True, False]))

        wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
        for wset in wyckoff_sets_conv:
            if wset.element == "Mo":
                self.assertEqual(wset.wyckoff_letter, "a")
            if wset.element == "S":
                self.assertEqual(wset.wyckoff_letter, "h")

    def test_mos2_vacuum(self):
        """Tests a non-flat 2D system with vacuum.
        """
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(5, 5, 1),
            vacuum=8
        )

        analyzer = SymmetryAnalyzer(system)
        conv_system = analyzer.get_conventional_system()
        prim_system = analyzer.get_primitive_system()
        self.assertTrue(np.array_equal(conv_system.get_pbc(), [True, True, False]))
        self.assertTrue(np.array_equal(prim_system.get_pbc(), [True, True, False]))

        wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
        for wset in wyckoff_sets_conv:
            if wset.element == "Mo":
                self.assertEqual(wset.wyckoff_letter, "a")
            if wset.element == "S":
                self.assertEqual(wset.wyckoff_letter, "h")


class WyckoffTests(unittest.TestCase):
    """Tests for the Wyckoff information.
    """
    def test_default_11(self):
        """See issue: https://github.com/spglib/spglib/issues/100
        """
        spg_11 = Atoms(
            symbols=["Cs", "Cs", "Cs", "Cs", "Hg", "Hg", "I", "I", "I", "I", "I", "I", "I", "I"],
            positions=np.array([
                [9.69146609791823e-10, 1.4864766792606409e-10, 2.1853577172462063e-10],
                [3.3905490010486266e-10, 4.918238896132229e-10, 2.21360430203356e-10],
                [-1.2127900679182296e-10, 5.914386740739359e-10, 6.432402437246207e-10],
                [5.088127028951374e-10, 2.4826245238677706e-10, 6.46064902203356e-10],
                [2.8716666854716376e-10, 6.364047202750034e-11, 2.173113402377552e-10],
                [5.607009344528362e-10, 6.764458699724996e-10, 6.420158122377551e-10],
                [6.774950241049414e-10, 5.508695960016385e-10, 2.0178558873663998e-11],
                [5.65784431911122e-10, 1.376848453015636e-10, 2.1575858671176543e-10],
                [-4.08306506008426e-11, 5.351034259842179e-10, 2.1998387806870958e-10],
                [6.805724578658182e-10, 5.50411253982781e-10, 4.175063512681291e-10],
                [1.7037257889505867e-10, 1.892167459983615e-10, 4.44883030873664e-10],
                [2.82083171088878e-10, 6.024014966984365e-10, 6.404630587117654e-10],
                [8.886982536008425e-10, 2.0498291601578212e-10, 6.446883500687096e-10],
                [1.6729514513418176e-10, 1.8967508801721902e-10, 8.42210823268129e-10]
            ])*1e10,
            cell=np.array([
                [1.132741769e-9, -1.0246116e-11, 0.0],
                [-2.84874166e-10,7.50332458e-10, 0.0],
                [0.0, 0.0, 8.49408944e-10]
            ])*1e10,
            pbc=True
        )

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(spg_11, symmetry_tol=0.1)
        spg = analyzer.get_space_group_number()
        self.assertEqual(spg, 11)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional(return_parameters=True)

    def test_default_87(self):
        """System where the equivalent_atoms reported by spglib do not match
        the Wyckoff sets in the standardized conventional cell. Must use
        crystallographic_orbits instead.
        """
        spg_87 = Atoms(
            symbols=[28, 28, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 38, 38, 38, 38, 52, 52],
            scaled_positions=[
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [5.00000783e-01, 5.00000105e-01, 5.00000504e-01],
                [2.57675031e-01, 2.57675031e-01, 0.00000000e+00],
                [7.42324606e-01, 7.42324606e-01, 0.00000000e+00],
                [7.42441461e-01, 2.57559427e-01, 4.18791129e-02],
                [4.57816602e-01, 5.42184286e-01, 2.42498450e-01],
                [4.13846169e-02, 9.58616271e-01, 2.57474206e-01],
                [7.57373910e-01, 2.42626978e-01, 4.58800327e-01],
                [2.42314486e-01, 2.42313809e-01, 5.00000504e-01],
                [7.57685827e-01, 7.57685150e-01, 5.00000504e-01],
                [2.42627655e-01, 7.57373233e-01, 5.41200681e-01],
                [9.58616953e-01, 4.13839350e-02, 7.42525529e-01],
                [5.42183696e-01, 4.57817192e-01, 7.57501282e-01],
                [2.57560109e-01, 7.42440779e-01, 9.58120623e-01],
                [9.99745256e-01, 5.00077432e-01, 2.50229920e-01],
                [4.99921738e-01, 2.53959000e-04, 2.50230128e-01],
                [5.00078114e-01, 9.99744574e-01, 7.49769816e-01],
                [2.54636000e-04, 4.99921061e-01, 7.49770880e-01],
                [5.00000444e-01, 5.00000444e-01, 0.00000000e+00],
                [3.38500000e-07, 9.99999661e-01, 5.00000504e-01],
            ],
            cell=[
                [-3.93057, -3.994236, -0.010812],
                [3.93057, -3.994236, 0.010812],
                [-0.030735, 0.0, 7.861732]
            ],
            pbc=True
        )

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(spg_87)
        spg = analyzer.get_space_group_number()
        self.assertEqual(spg, 87)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that groups are correct
        expected_sets = [
            WyckoffSet("a", 28, "Ni", multiplicity=2, space_group=87),
            WyckoffSet("b", 52, "Te", multiplicity=2, space_group=87),
            WyckoffSet("d", 38, "Sr", multiplicity=4, space_group=87),
            WyckoffSet("e", 8, "O", multiplicity=4, space_group=87, z=0.7423193187499998),
            WyckoffSet("h", 8, "O", multiplicity=8, space_group=87, x=0.78407358945, y=0.70099429955),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_non_default_160(self):
        """Tests a very long system where the position detection may fail if
        precicions are too strict or wrapping is done incorrectly.
        """
        sg_160 = Atoms(
            symbols=['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn', 'Zn'],
            scaled_positions = [
                [0.33796749, 0.66203193, 0.0139048],
                [0.02316237, 0.97683878, 0.06948712],
                [0.70834082, 0.29166138, 0.12501806],
                [0.39352094, 0.60647848, 0.18056513],
                [0.74538519, 0.25461701, 0.23615118],
                [0.09722873, 0.90277243, 0.29168619],
                [0.44907296, 0.55092646, 0.34722121],
                [0.1342672,  0.86573396, 0.40280159],
                [0.48612271, 0.51387671, 0.45837045],
                [0.83797036, 0.16203183, 0.5139067],
                [0.18981284, 0.81018832, 0.56943853],
                [0.87500806, 0.12499413, 0.62501979],
                [0.56018782, 0.43981159, 0.68056579],
                [0.24537727, 0.75462388, 0.73613182],
                [0.9305605,  0.06944169, 0.79167711],
                [0.61574641, 0.38425301, 0.84724154],
                [0.3009293,  0.69907186, 0.9027879],
                [0.98611365, 0.01388855, 0.95833655],
                [0.,         0.,         0.        ],
                [0.35185321, 0.64814621, 0.05556195],
                [0.03704207, 0.96295909, 0.1111262],
                [0.72222199, 0.27778021, 0.16666157],
                [0.40741297, 0.59258644, 0.22224124],
                [0.75926317, 0.24073903, 0.27778512],
                [0.11111061, 0.88889055, 0.33333183],
                [0.46296104, 0.53703838, 0.38888545],
                [0.14815321, 0.85184795, 0.44445962],
                [0.50000157, 0.49999784, 0.50000704],
                [0.85185135, 0.14815084, 0.55554968],
                [0.20370281, 0.79629834, 0.61110844],
                [0.88889041, 0.11111178, 0.66666684],
                [0.57407189, 0.42592753, 0.72221798],
                [0.25926051, 0.74074065, 0.77778153],
                [0.94444391, 0.05555828, 0.83332735],
                [0.62963,    0.37036942, 0.88889231],
                [0.31481059, 0.68519057, 0.94443176]
            ],
            cell=[[-1.917818, 3.321758, 0.0], [3.835636, 0.0, 0.0], [1.917818, -1.107253, -56.433045]],
            pbc=True
        )

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(sg_160)
        spg = analyzer.get_space_group_number()
        self.assertEqual(spg, 160)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

    def test_non_default_68(self):
        """Tests that systems which deviate from the default settings (spglib
        does not use the default settings, but instead will use the setting
        with lowest Hall number) are handled correctly.
        """
        sg_68 = Atoms(
            symbols=["Au", "Au", "Sn", "Sn", "Sn", "Sn", "Sn", "Sn", "Sn", "Sn"],
            scaled_positions=[
                [0.9852540807, 0.0000000000, 0.9926270404],
                [0.9852540807, 0.5000004882, 0.4926265201],
                [0.7215549731, 0.3327377765, 0.0235395469],
                [0.7215549731, 0.8327367459, 0.1980144280],
                [0.2456177163, 0.6668151491, 0.2855491427],
                [0.2456177163, 0.1668146609, 0.4600695715],
                [0.7215549731, 0.1672627118, 0.5235405450],
                [0.7215549731, 0.6672632000, 0.6980154261],
                [0.2456177163, 0.8331847967, 0.7855486225],
                [0.2456177163, 0.3331858273, 0.9600690513],
            ],
            cell= [
                [0.000000, -3.293253, 5.939270],
                [6.584074, 0.000000, 0.000000],
                [0.000000, 6.586507, 0.000000],
            ],
            pbc=True
        )

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(sg_68)
        spg = analyzer.get_space_group_number()
        self.assertEqual(spg, 68)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that groups are correct
        expected_sets = [
            WyckoffSet("a", 79, "Au", space_group=68, multiplicity=4),
            WyckoffSet("i", 50, "Sn", x=0.16276206029999996, y=0.11815044619999993, z=0.5827377765000001, space_group=68, multiplicity=16),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_non_default_129(self):
        """Tests that systems which deviate from the default settings (spglib
        does not use the default settings, but instead will use the setting
        with lowest Hall number) are handled correctly.
        """
        sg_129 = Atoms(
            symbols=["F", "F", "Nd", "Nd", "S", "S"],
            scaled_positions=[
                [0.7499993406, 0.2499997802, 0.0000000000],
                [0.2499997802, 0.7499993406, 0.0000000000],
                [0.2499997802, 0.2499997802, 0.2301982694],
                [0.7499993406, 0.7499993406, 0.7698006940],
                [0.7499993406, 0.7499993406, 0.3524397980],
                [0.2499997802, 0.2499997802, 0.6475606156],
            ],
            cell= [
                [3.919363, 0.000000, 0.000000],
                [0.000000, 3.919363, 0.000000],
                [0.000000, 0.000000, 6.895447],
            ],
            pbc=True
        )

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(sg_129)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()
        spg = analyzer.get_space_group_number()
        self.assertEqual(spg, 129)

        # Check that groups are correct
        expected_sets = [
            WyckoffSet("a", 9, "F", space_group=129, multiplicity=2),
            WyckoffSet("c", 16, "S", z=0.352439798, space_group=129, multiplicity=2),
            WyckoffSet("c", 60, "Nd", z=0.7698017306, space_group=129, multiplicity=2),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_default_225(self):
        """Tests that systems which deviate from the default settings (spglib
        does not use the default settings, but instead will use the setting
        with lowest Hall number) are handled correctly.
        """
        # Create structure that has space group 129: the origin setting differ
        # from default settings.
        a = 2.87
        system = ase.spacegroup.crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(system)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that groups are correct
        expected_sets = [
            WyckoffSet("a", 13, "Al", space_group=225, multiplicity=4),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_default_194(self):
        """This space group has a bit more complex expressions for the
        positions.
        """
        sg_194 = Atoms(
            symbols=['As', 'As', 'As', 'As', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'Na', 'Na', 'Na', 'Na', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Ru', 'Ru', 'Ru', 'Ru'],
            scaled_positions=[
                [0.33333199, 0.66666399, 0.05175819],
                [0.33333199, 0.66666399, 0.44824203],
                [0.66666821, 0.33333595, 0.55175807],
                [0.66666821, 0.33333595, 0.9482419 ],
                [0.,         0.,         0.,       ],
                [0.66666722, 0.33333397, 0.09251205],
                [0.33333397, 0.66666795, 0.1788528 ],
                [0.,         0.,         0.24999994],
                [0.33333397, 0.66666795, 0.32114741],
                [0.66666722, 0.33333397, 0.40748782],
                [0.,         0.,         0.49999988],
                [0.33333298, 0.66666597, 0.59251227],
                [0.66666624, 0.33333199, 0.67885268],
                [0.,         0.,         0.75000015],
                [0.66666624, 0.33333199, 0.82114729],
                [0.33333298, 0.66666597, 0.9074877 ],
                [0.,         0.,         0.12337398],
                [0.,         0.,         0.37662623],
                [0.,         0.,         0.62337386],
                [0.,         0.,         0.87662611],
                [0.6666791,  0.33335772, 0.00555923],
                [0.17488078, 0.82512516, 0.07300426],
                [0.65024438, 0.82512516, 0.07300426],
                [0.17487775, 0.34975551, 0.07300762],
                [0.34937053, 0.17468681, 0.1694312 ],
                [0.82531675, 0.17468681, 0.1694312 ],
                [0.82531439, 0.65062831, 0.16943187],
                [0.51194983, 0.02389918, 0.24999994],
                [0.51194863, 0.48805335, 0.24999994],
                [0.97610519, 0.48805335, 0.24999994],
                [0.82531439, 0.65062831, 0.330568  ],
                [0.34937053, 0.17468681, 0.33056868],
                [0.82531675, 0.17468681, 0.33056868],
                [0.17487775, 0.34975551, 0.42699259],
                [0.17488078, 0.82512516, 0.42699596],
                [0.65024438, 0.82512516, 0.42699596],
                [0.6666791,  0.33335772, 0.49444065],
                [0.33332111, 0.66664222, 0.50555911],
                [0.34975412, 0.17487479, 0.57300413],
                [0.82511943, 0.17487479, 0.57300413],
                [0.82512246, 0.65024444, 0.5730075 ],
                [0.17468346, 0.82531314, 0.66943108],
                [0.65062968, 0.82531314, 0.66943108],
                [0.17468582, 0.34937163, 0.66943175],
                [0.02389502, 0.51194659, 0.75000015],
                [0.48805157, 0.51194659, 0.75000015],
                [0.48805038, 0.97610076, 0.75000015],
                [0.17468582, 0.34937163, 0.83056822],
                [0.17468346, 0.82531314, 0.83056889],
                [0.65062968, 0.82531314, 0.83056889],
                [0.82512246, 0.65024444, 0.92699247],
                [0.34975412, 0.17487479, 0.92699584],
                [0.82511943, 0.17487479, 0.92699584],
                [0.33332111, 0.66664222, 0.99444086],
                [0.6666692,  0.33333793, 0.20364914],
                [0.6666692,  0.33333793, 0.29635074],
                [0.33333101, 0.66666201, 0.70364901],
                [0.33333101, 0.66666201, 0.79635095],
            ],
            cell=[[5.835617, 0.0, 0.0], [-2.917809, 5.05373, 0.0], [0.0, 0.0, 29.701967]],
            pbc=True,
        )

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(sg_194)
        spg = analyzer.get_space_group_number()
        self.assertEqual(spg, 194)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that groups are correct
        expected_sets = [
            WyckoffSet("a", 56, "Ba", multiplicity=2, space_group=194),
            WyckoffSet("b", 56, "Ba", multiplicity=2, space_group=194),
            WyckoffSet("e", 11, "Na", multiplicity=4, space_group=194, z=0.12337398),
            WyckoffSet("f", 8, "O", multiplicity=4, space_group=194, z=0.50555923),
            WyckoffSet("f", 33, "As", multiplicity=4, space_group=194, z=0.05175819),
            WyckoffSet("f", 44, "Ru", multiplicity=4, space_group=194, z=0.70364914),
            WyckoffSet("f", 56, "Ba", multiplicity=4, space_group=194, z=0.59251205),
            WyckoffSet("f", 56, "Ba", multiplicity=4, space_group=194, z=0.1788528),
            WyckoffSet("h", 8, "O", multiplicity=6, space_group=194, x=0.5119494849999999),
            WyckoffSet("k", 8, "O", multiplicity=12, space_group=194, x=0.17487978999999995, z=0.07300426000000003),
            WyckoffSet("k", 8, "O", multiplicity=12, space_group=194, x=0.82531286, z=0.1694312),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_no_free(self):
        """Test that no Wyckoff parameter is returned when the position is not
        free.
        """
        # Create structure
        a = 2.87
        fcc = ase.spacegroup.crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that the information matches
        self.assertEqual(len(wyckoff_sets), 1)
        wset = wyckoff_sets[0]
        self.assertEqual(wset.atomic_number, 13)
        self.assertEqual(wset.element, "Al")
        self.assertEqual(wset.wyckoff_letter, "a")
        self.assertEqual(wset.indices, [0, 1, 2, 3])
        self.assertEqual(wset.x, None)
        self.assertEqual(wset.y, None)
        self.assertEqual(wset.z, None)

    def test_one_free(self):
        """Test finding the value of the Wyckoff free parameter when one
        direction is free.
        """
        # Create structure
        var = {"x": 0.13}
        a = 12
        fcc = ase.spacegroup.crystal('Al', [(0, var["x"], 0)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        norm_sys = analyzer.get_conventional_system()
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that the information matches
        expected_sets = [
            WyckoffSet("e", 13, "Al", space_group=225, multiplicity=24, x=var["x"]),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_two_free(self):
        """Test finding the value of the Wyckoff free parameter when two
        directions are free.
        """
        # Create structure
        var = {"y": 0.077, "z": 0.13}
        a = 40
        fcc = ase.spacegroup.crystal('Al', [(0, var["y"], var["z"])], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that the information matches. The Wyckoff letters may not match
        # the ones that are used in creating the structure, but they are
        # nonetheless consistently determined.
        expected_sets = [
            WyckoffSet("j", 13, "Al", space_group=225, multiplicity=96, y=0.077, z=0.87),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)

    def test_three_free(self):
        """Test finding the value of the Wyckoff free parameter when three
        directions are free.
        """
        # Create structure
        var = {"x": 0.077, "y": 0.13, "z": 0.36}
        a = 100
        fcc = ase.spacegroup.crystal('Al', [(var["x"], var["y"], var["z"])], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])

        # Find the Wyckoff groups
        analyzer = SymmetryAnalyzer(fcc)
        wyckoff_sets = analyzer.get_wyckoff_sets_conventional()

        # Check that the information matches. The Wyckoff letters may not match
        # the ones that are used in creating the structure, but they are
        # nonetheless consistently determined.
        expected_sets = [
            WyckoffSet("l", 13, "Al", space_group=225, multiplicity=192, x=0.577, y=0.13, z=0.86),
        ]
        for w1, w2 in zip(expected_sets, wyckoff_sets):
            self.assertEqual(w1, w2)


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
        """This spacegroup (47) has additional affine normalizers that should
        not be taken into account when determining the conventional system.
        There are only 8 different possible Euclidean normalizations, whereas
        there are 48 normalizations if you take into account the affine ones as
        well.
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

    def test_zinc_blende(self):
        """Tests that all different forms of the zinc-blende structure can be
        normalized to the same structure. In order to do this properly we need
        to also include Euclidean normalizers where the transformation is not a
        proper rigid rotations, but due to the symmetry of the structure result
        in a valid transformation.
        """
        # Primitive
        zb_prim = ase.build.bulk("ZnS", crystalstructure="zincblende", a=5.42)
        analyzer_prim = SymmetryAnalyzer(zb_prim)
        zb_prim_conv = analyzer_prim.get_conventional_system()
        wyckoff_prim = analyzer_prim.get_wyckoff_letters_conventional()
        pos_prim = zb_prim_conv.get_positions()
        z_prim = zb_prim_conv.get_atomic_numbers()

        # Conventional
        zb_conv = ase.build.bulk("ZnS", crystalstructure="zincblende", a=5.42, cubic=True)
        analyzer_conv = SymmetryAnalyzer(zb_conv)
        zb_conv_conv = analyzer_conv.get_conventional_system()
        wyckoff_conv = analyzer_conv.get_wyckoff_letters_conventional()
        pos_conv = zb_conv_conv.get_positions()
        z_conv = zb_conv_conv.get_atomic_numbers()

        # Assume that the same atoms are at same positions
        for ipos, iz, iw in zip(pos_prim, z_prim, wyckoff_prim):
            found = False
            for jpos, jz, jw in zip(pos_prim, z_prim, wyckoff_prim):
                match_pos = np.allclose(ipos, jpos)
                match_z = iz == jz
                match_w = iw == jw
                if match_pos and match_z and match_w:
                    found = True
            self.assertTrue(found)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestSegfaultProtect))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyser3DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SymmetryAnalyser2DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(WyckoffTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GroundStateTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
