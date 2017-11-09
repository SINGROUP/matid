"""
Defines a set of regressions tests that should be run succesfully before
anything is pushed to the central repository.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import sys

import numpy as np

from ase import Atoms
from ase.build import bcc100, add_adsorbate, molecule
from ase.visualize import view
import ase.build
from ase.build import nanotube
import ase.lattice.hexagonal

from systax import Classifier
from systax.classification import Atom, Molecule, Crystal, Material1D, Material2D, Unknown, Surface, AdsorptionSystem
from systax import Material3DAnalyzer
from systax.data.constants import WYCKOFF_LETTER_POSITIONS
import systax.geometry


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class GeometryTests(unittest.TestCase):
    """Tests for the geometry module.
    """
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

        disp_tensor = systax.geometry.get_displacement_tensor(pos1, pos2)
        expected = np.array(-pos2)
        self.assertTrue(np.allclose(disp_tensor, expected))

        # Fully periodic
        disp_tensor = systax.geometry.get_displacement_tensor(pos1, pos2, pbc=True, cell=cell)
        expected = np.array([[
            [0, 0, 0],
            [0.1, 0, 0],
        ]])
        self.assertTrue(np.allclose(disp_tensor, expected))

        # Fully periodic, reversed direction
        disp_tensor = systax.geometry.get_displacement_tensor(pos2, pos1, pbc=True, cell=cell)
        expected = np.array([[
            [0, 0, 0],
        ], [
            [-0.1, 0, 0],
        ]])
        self.assertTrue(np.allclose(disp_tensor, expected))

        # Periodic in one direction
        disp_tensor = systax.geometry.get_displacement_tensor(pos1, pos2, pbc=[True, False, False], cell=cell)
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
        cart_pos = systax.geometry.to_cartesian(cell, rel_pos)
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
        cart_pos = systax.geometry.to_cartesian(cell, rel_pos)
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
        cart_pos = systax.geometry.to_cartesian(cell, rel_pos, wrap=True, pbc=True)
        self.assertTrue(np.allclose(cart_pos, expected_pos))


class AtomTests(unittest.TestCase):
    """Tests for detecting an Atom.
    """
    def test_finite(self):
        classifier = Classifier()
        c = Atoms(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]), pbc=False)
        clas = classifier.classify(c)
        self.assertIsInstance(clas, Atom)

    def test_periodic(self):
        classifier = Classifier()
        c = Atoms(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]), pbc=True, cell=[10, 10, 10])
        clas = classifier.classify(c)
        self.assertIsInstance(clas, Atom)

        c = Atoms(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]), pbc=[1, 0, 1], cell=[10, 10, 10])
        clas = classifier.classify(c)
        self.assertIsInstance(clas, Atom)

        c = Atoms(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]), pbc=[1, 0, 0], cell=[10, 10, 10])
        clas = classifier.classify(c)
        self.assertIsInstance(clas, Atom)


class MoleculeTests(unittest.TestCase):
    """Tests for detecting a molecule.
    """
    def test_h2o_no_pbc(self):
        h2o = molecule("H2O")
        classifier = Classifier()
        clas = classifier.classify(h2o)
        self.assertIsInstance(clas, Molecule)

    def test_h2o_pbc(self):
        h2o = molecule("H2O")
        gap = 10
        h2o.set_cell([[gap, 0, 0], [0, gap, 0], [0, 0, gap]])
        h2o.set_pbc([True, True, True])
        h2o.center()
        classifier = Classifier()
        clas = classifier.classify(h2o)
        self.assertIsInstance(clas, Molecule)


class Material1DTests(unittest.TestCase):
    """Tests detection of bulk 3D materials.
    """
    def test_nanotube_full_pbc(self):
        tube = nanotube(6, 0, length=1)
        tube.set_pbc([True, True, True])
        cell = tube.get_cell()
        cell[0][0] = 20
        cell[1][1] = 20
        tube.set_cell(cell)
        tube.center()

        classifier = Classifier()
        clas = classifier.classify(tube)
        self.assertIsInstance(clas, Material1D)

    def test_nanotube_partial_pbc(self):
        tube = nanotube(6, 0, length=1)
        tube.set_pbc([False, False, True])
        cell = tube.get_cell()
        cell[0][0] = 6
        cell[1][1] = 6
        tube.set_cell(cell)
        tube.center()

        classifier = Classifier()
        clas = classifier.classify(tube)
        self.assertIsInstance(clas, Material1D)

    def test_nanotube_full_pbc_shaken(self):
        tube = nanotube(6, 0, length=1)
        tube.set_pbc([True, True, True])
        cell = tube.get_cell()
        cell[0][0] = 20
        cell[1][1] = 20
        tube.set_cell(cell)
        tube.rattle(0.1, seed=42)
        tube.center()

        classifier = Classifier()
        clas = classifier.classify(tube)
        self.assertIsInstance(clas, Material1D)

    def test_nanotube_too_big(self):
        """Test that too big 1D structures are classifed as unknown.
        """
        tube = nanotube(20, 0, length=1)
        tube.set_pbc([True, True, True])
        cell = tube.get_cell()
        cell[0][0] = 40
        cell[1][1] = 40
        tube.set_cell(cell)
        tube.center()

        classifier = Classifier()
        clas = classifier.classify(tube)
        self.assertIsInstance(clas, Unknown)


class Material2DTests(unittest.TestCase):
    """Tests detection of bulk 3D materials.
    """
    # def test_graphene_primitive(self):

        # graphene = Atoms(
            # symbols=[6, 6],
            # cell=np.array((
                # [
                    # 2.4595121467478055,
                    # 0.0,
                    # 0.0
                # ],
                # [
                    # -1.2297560733739028,
                    # 2.13,
                    # 0.0
                # ],
                # [
                    # 0.0,
                    # 0.0,
                    # 20.0
                # ]
            # )),
            # scaled_positions=np.array((
                # [
                    # 0.3333333333333333,
                    # 0.6666666666666666,
                    # 0.5
                # ],
                # [
                    # 0.6666666666666667,
                    # 0.33333333333333337,
                    # 0.5
                # ]
            # )),
            # pbc=True
        # )
        # # view(graphene)

        # classifier = Classifier()
        # clas = classifier.classify(graphene)
        # self.assertIsInstance(clas, Material2D)

    def test_graphene_supercell(self):
        graphene = Atoms(
            symbols=[6, 6],
            cell=np.array((
                [
                    2.4595121467478055,
                    0.0,
                    0.0
                ],
                [
                    -1.2297560733739028,
                    2.13,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    20.0
                ]
            )),
            scaled_positions=np.array((
                [
                    0.3333333333333333,
                    0.6666666666666666,
                    0.5
                ],
                [
                    0.6666666666666667,
                    0.33333333333333337,
                    0.5
                ]
            )),
            pbc=True
        )

        graphene = graphene.repeat([5, 5, 1])
        # view(graphene)

        classifier = Classifier()
        clas = classifier.classify(graphene)
        # self.assertIsInstance(clas, Material2D)

    # def test_graphene_partial_pbc(self):
        # graphene = Atoms(
            # symbols=[6, 6],
            # cell=np.array((
                # [
                    # 2.4595121467478055,
                    # 0.0,
                    # 0.0
                # ],
                # [
                    # -1.2297560733739028,
                    # 2.13,
                    # 0.0
                # ],
                # [
                    # 0.0,
                    # 0.0,
                    # 1.0
                # ]
            # )),
            # scaled_positions=np.array((
                # [
                    # 0.3333333333333333,
                    # 0.6666666666666666,
                    # 0.5
                # ],
                # [
                    # 0.6666666666666667,
                    # 0.33333333333333337,
                    # 0.5
                # ]
            # )),
            # pbc=[True, True, False]
        # )

        # classifier = Classifier()
        # clas = classifier.classify(graphene)
        # self.assertIsInstance(clas, Material2D)


class Material3DTests(unittest.TestCase):
    """Tests detection of bulk 3D materials.
    """
    def test_si(self):
        si = ase.lattice.cubic.Diamond(
            size=(1, 1, 1),
            symbol='Si',
            pbc=(1, 1, 1),
            latticeconstant=5.430710)
        classifier = Classifier()
        clas = classifier.classify(si)
        self.assertIsInstance(clas, Crystal)

    def test_graphite(self):
        """Testing a sparse material like graphite.
        """
        sys = ase.lattice.hexagonal.Graphite(
            size=(1, 1, 1),
            symbol='C',
            pbc=(1, 1, 1),
            latticeconstant=(2.461, 6.708))
        classifier = Classifier()
        clas = classifier.classify(sys)
        self.assertIsInstance(clas, Crystal)

    def test_amorphous(self):
        """Test an amorphous crystal with completely random positions. This is
        currently not classified as crystal, but the threshold can be set in
        the classifier setup.
        """
        np.random.seed(8)
        n_atoms = 50
        rand_pos = np.random.rand(n_atoms, 3)

        sys = Atoms(
            scaled_positions=rand_pos,
            cell=(10, 10, 10),
            symbols=n_atoms*['C'],
            pbc=(1, 1, 1))
        classifier = Classifier()
        clas = classifier.classify(sys)
        self.assertIsInstance(clas, Unknown)

    def test_too_sparse(self):
        """Test a crystal that is too sparse.
        """
        sys = ase.lattice.hexagonal.Graphite(
            size=(1, 1, 1),
            symbol='C',
            pbc=(1, 1, 1),
            latticeconstant=(2.461, 10))
        # view(sys)
        classifier = Classifier()
        clas = classifier.classify(sys)
        self.assertIsInstance(clas, Unknown)


class Material3DAnalyserTests(unittest.TestCase):
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
        a *= 1.04
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
        )

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
        self.assertTrue(np.array_equal(data.equivalent_original, [0, 1]))
        self.assertTrue(np.array_equal(data.wyckoff_original, ["a", "b"]))
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
        np.random.seed(seed=42)
        positions = 10*np.random.rand(10, 3)
        system = Atoms(
            positions=positions,
            symbols=["H", "C", "Na", "Fe", "Cu", "He", "Ne", "Mg", "Si", "Ti"],
            cell=[10, 10, 10]
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
        for (i_w, i_z), group_list in wyckoff_groups.items():

            # Check that the current Wyckoff letter index is greater than
            # previous, if not the atomic number must be greater
            i_w_index = WYCKOFF_LETTER_POSITIONS[i_w]
            if prev_w_index is not None:
                self.assertGreaterEqual(i_w_index, prev_w_index)
                if i_w_index == prev_w_index:
                    self.assertGreater(i_z, prev_z)

            prev_w_index = i_w_index
            prev_z = i_z

            # Gather the number of atoms in eaach group to see that it matches
            # the amount of atoms in the system
            for group in group_list:
                n = len(group.positions)
                n_atoms_wyckoff += n

        self.assertEqual(n_atoms, n_atoms_wyckoff)

    def get_material3d_properties(self, system):
        analyzer = Material3DAnalyzer(system)
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


class SurfaceTests(unittest.TestCase):
    """Tests for detecting and analyzing surfaces.
    """
    def test_perfect_surface(self):

        # Create an Fe 100 surface as an ASE Atoms object
        system = bcc100('Fe', size=(3, 3, 3), vacuum=8)
        view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Test surface info
        # # surfaces = classifier.surfaces
        # surface = classification.surfaces[0]
        # analyzer = surface.bulk_analyzer
        # bulk_system = analyzer.get_conventional_system()
        # pos = bulk_system.get_scaled_positions()
        # numbers = bulk_system.get_atomic_numbers()
        # cell = bulk_system.get_cell()
        # space_group = analyzer.get_symmetry_dataset()["number"]
        # wyckoffs = analyzer.get_wyckoff_letters_conventional()
        # surf_ind = set(surface.indices)
        # expected_surf_ind = set(range(len(system)))

        # cell_expected = np.array([[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]])
        # pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        # self.assertEqual(len(classification.surfaces), 1)
        # self.assertEqual(surf_ind, expected_surf_ind)
        # self.assertEqual(space_group, 229)
        # self.assertTrue(np.array_equal(numbers, [26, 26]))
        # self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        # self.assertTrue(np.array_equal(pos, pos_expected))
        # self.assertTrue(np.allclose(cell, cell_expected))

        # Check that we find the correct surface atoms
        # surf_analyzer = surface.analyzer()
        # top_atoms = surf_analyzer.get_top_indices()

    # def test_adsorbate(self):

        # # Create an Fe 100 surface with a benzene adsorbate as an ASE Atoms
        # # object
        # system = bcc100('Fe', size=(3, 3, 3), vacuum=8)
        # mol = ase.build.molecule("C6H6")
        # add_adsorbate(system, mol, height=2, offset=(2, 2.5))
        # view(system)

        # classifier = Classifier()
        # classifier.classify(system)

        # # Test surface info
        # surfaces = classifier.surfaces
        # surface = surfaces[0]
        # bulk_system = surface.get_normalized_cell()
        # pos = bulk_system.relative_pos
        # numbers = bulk_system.numbers
        # cell = bulk_system.lattice.matrix
        # wyckoffs = bulk_system.wyckoff_letters
        # space_group = surface.symmetry_dataset["number"]
        # exp_mol_ind = set((27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38))
        # exp_surf_ind = set(range(len(system))) - exp_mol_ind
        # surf_ind = set(surface.indices)
        # cell_expected = np.array([[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]])
        # pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        # self.assertEqual(len(surfaces), 1)
        # self.assertEqual(surf_ind, exp_surf_ind)
        # self.assertEqual(space_group, 229)
        # self.assertTrue(np.array_equal(numbers, [26, 26]))
        # self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        # self.assertTrue(np.array_equal(pos, pos_expected))
        # self.assertTrue(np.allclose(cell, cell_expected))

        # # Test molecule info
        # molecules = classifier.molecules
        # molecule = molecules[0]
        # mol_ind = set(molecule.indices)
        # self.assertEqual(len(molecules), 1)
        # self.assertEqual(mol_ind, exp_mol_ind)

    # def test_imperfect_surface(self):

        # # Create an Fe 100 surface as an ASE Atoms object
        # system = bcc100('Fe', size=(12, 12, 3), vacuum=8)
        # system.rattle(stdev=0.15, seed=42)
        # view(system)

        # classifier = Classifier()
        # classifier.classify(system)

        # # Test surface info
        # surfaces = classifier.surfaces
        # surface = surfaces[0]
        # bulk_system = surface.get_normalized_cell()
        # pos = bulk_system.relative_pos
        # numbers = bulk_system.numbers
        # wyckoffs = bulk_system.wyckoff_letters
        # space_group = surface.symmetry_dataset["number"]
        # surf_ind = set(surface.indices)
        # expected_surf_ind = set(range(len(system)))

        # pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        # self.assertEqual(len(surfaces), 1)
        # self.assertEqual(surf_ind, expected_surf_ind)
        # self.assertEqual(space_group, 229)
        # self.assertTrue(np.array_equal(numbers, [26, 26]))
        # self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        # self.assertTrue(np.array_equal(pos, pos_expected))

    # def test_curved_surface(self):

        # # Create an Fe 100 surface as an ASE Atoms object
        # system = bcc100('Fe', size=(12, 12, 3), vacuum=8)
        # # system.rattle(stdev=0.0005, seed=42)

        # # Bulge the surface
        # cell_width = np.linalg.norm(system.get_cell()[0, :])
        # for atom in system:
            # pos = atom.position
            # distortion_z = 1.0*np.sin(pos[0]/cell_width*2.0*np.pi)
            # pos += np.array((0, 0, distortion_z))
        # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)

        # # Test surface info
        # surfaces = classification.surfaces
        # surface = surfaces[0]
        # bulk_analyzer = surface.bulk_analyzer
        # bulk_system = bulk_analyzer.get_conventional_system()
        # pos = bulk_system.get_scaled_positions()
        # numbers = bulk_system.get_atomic_numbers()
        # wyckoffs = bulk_system.get_wyckoff_letters()
        # space_group = bulk_analyzer.get_space_group_number()
        # surf_ind = set(surface.indices)
        # expected_surf_ind = set(range(len(system)))

        # pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        # self.assertEqual(len(surfaces), 1)
        # self.assertEqual(surf_ind, expected_surf_ind)
        # self.assertEqual(space_group, 229)
        # self.assertTrue(np.array_equal(numbers, [26, 26]))
        # self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        # self.assertTrue(np.array_equal(pos, pos_expected))

if __name__ == '__main__':
    suites = []
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(GeometryTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(AtomTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(MoleculeTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(Material1DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Material2DTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(Material3DTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(Material3DAnalyserTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(BCCTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
