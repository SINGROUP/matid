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

from systax import Classifier
from systax.classification import Atom, Molecule, Crystal, Material1D, Material2D, Unknown
from systax import Material3DAnalyzer


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
    def test_graphene_full_pbc(self):

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

        classifier = Classifier()
        clas = classifier.classify(graphene)
        self.assertIsInstance(clas, Material2D)

    def test_graphene_partial_pbc(self):
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
                    1.0
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
            pbc=[True, True, False]
        )

        classifier = Classifier()
        clas = classifier.classify(graphene)
        self.assertIsInstance(clas, Material2D)


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


class Material3DAnalyserTests(unittest.TestCase):
    """Tests the analysis of bulk 3D materials.
    """
    def test_si(self):
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

        analyzer = Material3DAnalyzer(si)
        space_group_number = analyzer.get_space_group_number()
        space_group_int = analyzer.get_space_group_international_short()
        hall_symbol = analyzer.get_hall_symbol()
        hall_number = analyzer.get_hall_number()

        conv_system = analyzer.get_conventional_system()
        prim_system = analyzer.get_primitive_system()

        translations = analyzer.get_translations()
        rotations = analyzer.get_rotations()
        origin_shift = analyzer.get_origin_shift()
        choice = analyzer.get_choice()
        point_group = analyzer.get_point_group()
        transformation_matrix = analyzer.get_transformation_matrix()

        wyckoff_original = analyzer.get_wyckoff_letters_original()
        wyckoff_conv = analyzer.get_wyckoff_letters_conventional()

        prim_wyckoff = analyzer.get_wyckoff_letters_primitive()
        prim_equiv = analyzer.get_equivalent_atoms_primitive()

        equivalent_original = analyzer.get_equivalent_atoms_original()
        equivalent_conv = analyzer.get_equivalent_atoms_conventional()

        lattice_fit = analyzer.get_conventional_lattice_fit()

        # view(si)
        # view(conv_system)
        # view(prim_system)

        self.assertEqual(space_group_number, 227)
        self.assertEqual(space_group_int, "Fd-3m")
        self.assertEqual(hall_symbol, "F 4d 2 3 -1d")
        self.assertEqual(hall_number, 525)
        self.assertEqual(point_group, "m-3m")
        self.assertEqual(choice, "1")
        self.assertTrue(np.array_equal(equivalent_conv, [0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(wyckoff_conv, ["a", "a", "a", "a", "a", "a", "a", "a"]))
        self.assertTrue(np.array_equal(equivalent_original, [0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(wyckoff_original, ["a", "a", "a", "a", "a", "a", "a", "a"]))
        self.assertTrue(np.array_equal(prim_wyckoff, ["a", "a"]))
        self.assertTrue(np.array_equal(prim_equiv, [0, 0]))

        # Check that the lattice fit gets the volume and lattice vectors right
        n_atoms_orig = len(si)
        volume_orig = si.get_volume()
        n_atoms_conv = len(conv_system)
        volume_conv = np.linalg.det(lattice_fit)

        self.assertTrue(np.allclose(volume_orig/n_atoms_orig, volume_conv/n_atoms_conv, atol=1e-8))

    def test_nacl_primitive(self):
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

        analyzer = Material3DAnalyzer(nacl)
        space_group_number = analyzer.get_space_group_number()
        space_group_int = analyzer.get_space_group_international_short()
        hall_symbol = analyzer.get_hall_symbol()
        hall_number = analyzer.get_hall_number()

        conv_system = analyzer.get_conventional_system()
        prim_system = analyzer.get_primitive_system()

        translations = analyzer.get_translations()
        rotations = analyzer.get_rotations()
        origin_shift = analyzer.get_origin_shift()
        choice = analyzer.get_choice()
        point_group = analyzer.get_point_group()
        transformation_matrix = analyzer.get_transformation_matrix()

        wyckoff_original = analyzer.get_wyckoff_letters_original()
        wyckoff_conv = analyzer.get_wyckoff_letters_conventional()

        prim_wyckoff = analyzer.get_wyckoff_letters_primitive()
        prim_equiv = analyzer.get_equivalent_atoms_primitive()

        equivalent_original = analyzer.get_equivalent_atoms_original()
        equivalent_conv = analyzer.get_equivalent_atoms_conventional()

        lattice_fit = analyzer.get_conventional_lattice_fit()

        # view(nacl)
        # view(conv_system)
        # view(prim_system)

        self.assertEqual(space_group_number, 225)
        self.assertEqual(space_group_int, "Fm-3m")
        self.assertEqual(hall_symbol, "-F 4 2 3")
        self.assertEqual(hall_number, 523)
        self.assertEqual(point_group, "m-3m")
        self.assertEqual(choice, "")
        self.assertTrue(np.array_equal(equivalent_conv, [0, 1, 0, 1, 0, 1, 0, 1]))
        self.assertTrue(np.array_equal(wyckoff_conv, ["a", "b", "a", "b", "a", "b", "a", "b"]))
        self.assertTrue(np.array_equal(equivalent_original, [0, 1]))
        self.assertTrue(np.array_equal(wyckoff_original, ["a", "b"]))

        self.assertTrue(np.array_equal(prim_equiv, [0, 1]))
        self.assertTrue(np.array_equal(prim_wyckoff, ["a", "b"]))

        # Check that the lattice fit gets the volume right
        n_atoms_orig = len(nacl)
        volume_orig = nacl.get_volume()
        n_atoms_conv = len(conv_system)
        volume_conv = np.linalg.det(lattice_fit)

        self.assertTrue(np.allclose(volume_orig/n_atoms_orig, volume_conv/n_atoms_conv, atol=1e-8))

    def test_bcc(self):
        from ase.lattice.cubic import BodyCenteredCubic
        bcc = BodyCenteredCubic(
            directions=[[1, 0, 0], [0, 1, 0], [1, 1, 1]],
            size=(1, 1, 1),
            symbol='Cu',
            pbc=True,
            latticeconstant=4.0)

        analyzer = Material3DAnalyzer(bcc)
        space_group_number = analyzer.get_space_group_number()
        space_group_int = analyzer.get_space_group_international_short()
        hall_symbol = analyzer.get_hall_symbol()
        hall_number = analyzer.get_hall_number()

        conv_system = analyzer.get_conventional_system()
        prim_system = analyzer.get_primitive_system()

        translations = analyzer.get_translations()
        rotations = analyzer.get_rotations()
        origin_shift = analyzer.get_origin_shift()
        choice = analyzer.get_choice()
        point_group = analyzer.get_point_group()
        transformation_matrix = analyzer.get_transformation_matrix()

        wyckoff_original = analyzer.get_wyckoff_letters_original()
        wyckoff_conv = analyzer.get_wyckoff_letters_conventional()

        prim_wyckoff = analyzer.get_wyckoff_letters_primitive()
        prim_equiv = analyzer.get_equivalent_atoms_primitive()

        equivalent_original = analyzer.get_equivalent_atoms_original()
        equivalent_conv = analyzer.get_equivalent_atoms_conventional()

        lattice_fit = analyzer.get_conventional_lattice_fit()

        # view(nacl)
        # view(conv_system)
        # view(prim_system)

        # self.assertEqual(space_group_number, 225)
        # self.assertEqual(space_group_int, "Fm-3m")
        # self.assertEqual(hall_symbol, "-F 4 2 3")
        # self.assertEqual(hall_number, 523)
        # self.assertEqual(point_group, "m-3m")
        # self.assertEqual(choice, "")
        # self.assertTrue(np.array_equal(equivalent_conv, [0, 1, 0, 1, 0, 1, 0, 1]))
        # self.assertTrue(np.array_equal(wyckoff_conv, ["a", "b", "a", "b", "a", "b", "a", "b"]))
        # self.assertTrue(np.array_equal(equivalent_original, [0, 1]))
        # self.assertTrue(np.array_equal(wyckoff_original, ["a", "b"]))

        # # Check that the lattice fit gets the volume right
        # n_atoms_orig = len(nacl)
        # volume_orig = nacl.get_volume()
        # n_atoms_conv = len(conv_system)
        # volume_conv = np.linalg.det(lattice_fit)

        # self.assertTrue(np.allclose(volume_orig/n_atoms_orig, volume_conv/n_atoms_conv, atol=1e-8))

    def test_unsymmetric(self):
        np.random.seed(seed=42)
        positions = 10*np.random.rand(10, 3)
        atoms = Atoms(
            positions=positions,
            symbols=["C"]*10,
            cell=[10, 10, 10]
        )
        analyzer = Material3DAnalyzer(atoms)
        space_group_number = analyzer.get_space_group_number()
        conv_system = analyzer.get_conventional_system()
        prim_system = analyzer.get_primitive_system()

        # view(prim_system)


class BCCTests(unittest.TestCase):
    """Tests for a BCC crystal.
    """
    def test_perfect_surface(self):

        # Create an Fe 100 surface as an ASE Atoms object
        system = bcc100('Fe', size=(3, 3, 3), vacuum=8)
        view(system)

        classifier = Classifier()
        classifier.classify(system)

        # Test surface info
        surfaces = classifier.surfaces
        surface = surfaces[0]
        bulk_system = surface.get_normalized_cell()
        pos = bulk_system.relative_pos
        numbers = bulk_system.numbers
        cell = bulk_system.lattice.matrix
        wyckoffs = bulk_system.wyckoff_letters
        space_group = surface.symmetry_dataset["number"]
        surf_ind = set(surface.indices)
        expected_surf_ind = set(range(len(system)))

        cell_expected = np.array([[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]])
        pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surf_ind, expected_surf_ind)
        self.assertEqual(space_group, 229)
        self.assertTrue(np.array_equal(numbers, [26, 26]))
        self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        self.assertTrue(np.array_equal(pos, pos_expected))
        self.assertTrue(np.allclose(cell, cell_expected))

    def test_adsorbate(self):

        # Create an Fe 100 surface with a benzene adsorbate as an ASE Atoms
        # object
        system = bcc100('Fe', size=(3, 3, 3), vacuum=8)
        mol = ase.build.molecule("C6H6")
        add_adsorbate(system, mol, height=2, offset=(2, 2.5))
        view(system)

        classifier = Classifier()
        classifier.classify(system)

        # Test surface info
        surfaces = classifier.surfaces
        surface = surfaces[0]
        bulk_system = surface.get_normalized_cell()
        pos = bulk_system.relative_pos
        numbers = bulk_system.numbers
        cell = bulk_system.lattice.matrix
        wyckoffs = bulk_system.wyckoff_letters
        space_group = surface.symmetry_dataset["number"]
        exp_mol_ind = set((27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38))
        exp_surf_ind = set(range(len(system))) - exp_mol_ind
        surf_ind = set(surface.indices)
        cell_expected = np.array([[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]])
        pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surf_ind, exp_surf_ind)
        self.assertEqual(space_group, 229)
        self.assertTrue(np.array_equal(numbers, [26, 26]))
        self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        self.assertTrue(np.array_equal(pos, pos_expected))
        self.assertTrue(np.allclose(cell, cell_expected))

        # Test molecule info
        molecules = classifier.molecules
        molecule = molecules[0]
        mol_ind = set(molecule.indices)
        self.assertEqual(len(molecules), 1)
        self.assertEqual(mol_ind, exp_mol_ind)

    def test_imperfect_surface(self):

        # Create an Fe 100 surface as an ASE Atoms object
        system = bcc100('Fe', size=(3, 3, 3), vacuum=8)
        system.rattle(stdev=0.1, seed=42)
        view(system)

        classifier = Classifier()
        classifier.classify(system)

        # Test surface info
        surfaces = classifier.surfaces
        surface = surfaces[0]
        bulk_system = surface.get_normalized_cell()
        pos = bulk_system.relative_pos
        numbers = bulk_system.numbers
        wyckoffs = bulk_system.wyckoff_letters
        space_group = surface.symmetry_dataset["number"]
        surf_ind = set(surface.indices)
        expected_surf_ind = set(range(len(system)))

        pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surf_ind, expected_surf_ind)
        self.assertEqual(space_group, 229)
        self.assertTrue(np.array_equal(numbers, [26, 26]))
        self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        self.assertTrue(np.array_equal(pos, pos_expected))

    def test_curved_surface(self):

        # Create an Fe 100 surface as an ASE Atoms object
        system = bcc100('Fe', size=(12, 12, 3), vacuum=8)
        # system.rattle(stdev=0.0005, seed=42)

        # Bulge the surface
        cell_width = np.linalg.norm(system.get_cell()[0, :])
        for atom in system:
            pos = atom.position
            distortion_z = np.sin(pos[0]/cell_width*2*np.pi)
            pos += np.array((0, 0, distortion_z))
        view(system)

        classifier = Classifier()
        classifier.classify(system)

        # Test surface info
        surfaces = classifier.surfaces
        surface = surfaces[0]
        bulk_system = surface.get_normalized_cell()
        pos = bulk_system.relative_pos
        numbers = bulk_system.numbers
        wyckoffs = bulk_system.wyckoff_letters
        space_group = surface.symmetry_dataset["number"]
        surf_ind = set(surface.indices)
        expected_surf_ind = set(range(len(system)))

        pos_expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surf_ind, expected_surf_ind)
        self.assertEqual(space_group, 229)
        self.assertTrue(np.array_equal(numbers, [26, 26]))
        self.assertTrue(np.array_equal(wyckoffs, ["a", "a"]))
        self.assertTrue(np.array_equal(pos, pos_expected))

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(AtomTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MoleculeTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Material1DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Material2DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Material3DAnalyserTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(Material3DTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(BCCTests))

    import time
    alltests = unittest.TestSuite(suites)

    times = []
    for i in range(1):
        start = time.time()
        result = unittest.TextTestRunner(verbosity=0).run(alltests)
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
    print(np.array(times).mean())

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
