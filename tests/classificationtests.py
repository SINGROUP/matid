"""
Set of regression tests for structure classification.
"""
import unittest

import numpy as np
from numpy.random import RandomState

from ase import Atoms
from ase.build import bcc100, molecule
from ase.visualize import view
import ase.build
from ase.build import nanotube
import ase.lattice.hexagonal
from ase.lattice.compounds import Zincblende
from ase.lattice.cubic import SimpleCubicFactory
from ase.data import covalent_radii
import ase.io
from networkx import draw_networkx
import matplotlib.pyplot as mpl

from matid import Classifier, SymmetryAnalyzer, PeriodicFinder
from matid.classifications import \
    Class0D, \
    Class1D, \
    Class2D, \
    Class3D, \
    Atom, \
    Material2D, \
    Unknown, \
    Surface
import matid.geometry

from conftest import create_graphene


class ExceptionTests(unittest.TestCase):
    """Tests for exceptions that arise from invalid arguments.
    """
    def test_too_many_atoms(self):
        system = bcc100('Fe', size=(11, 10, 10), vacuum=8)

        classifier = Classifier()
        with self.assertRaises(ValueError):
            classifier.classify(system)


class RadiiTests(unittest.TestCase):
    """Tests that all available radii options are supported correctly.
    """
    # 2D system
    sys2d = Atoms(
        symbols=[6, 6],
        cell=np.array((
            [2.4595121467478055, 0.0, 0.0],
            [-1.2297560733739028, 2.13, 0.0],
            [0.0, 0.0, 10.0]
        )),
        scaled_positions=np.array((
            [0.3333333333333333, 0.6666666666666666, 0.5],
            [0.6666666666666667, 0.33333333333333337, 0.5]
        )),
        pbc=True
    )
    sys2d = sys2d.repeat((3, 3, 1))

    def test_covalent(self):
        classifier = Classifier(radii="covalent", cluster_threshold=1.5)
        clas = classifier.classify(RadiiTests.sys2d)
        self.assertIsInstance(clas, Class2D)

    def test_vdw(self):
        classifier = Classifier(radii="vdw", cluster_threshold=0.5)
        clas = classifier.classify(RadiiTests.sys2d)
        self.assertIsInstance(clas, Class2D)

    def test_vdw_covalent(self):
        classifier = Classifier(radii="vdw_covalent", cluster_threshold=0.5)
        clas = classifier.classify(RadiiTests.sys2d)
        self.assertIsInstance(clas, Class2D)

    def test_custom(self):
        classifier = Classifier(radii=0.5*covalent_radii, cluster_threshold=3)
        clas = classifier.classify(RadiiTests.sys2d)
        self.assertIsInstance(clas, Class2D)


class DimensionalityTests(unittest.TestCase):
    """Unit tests for finding the dimensionality of different systems.
    """
    # 0D system
    sys0d = molecule("H2O")
    sys0d.set_pbc([False, False, False])
    sys0d.set_cell([3, 3, 3])
    sys0d.center()

    # 1D system
    sys1d = nanotube(3, 3, length=6, bond=1.4, symbol='Si')
    sys1d.set_pbc([True, True, True])
    sys1d.set_cell((10, 10, 15))
    sys1d.center()

    # 2D system
    sys2d = Atoms(
        symbols=[6, 6],
        cell=np.array((
            [2.4595121467478055, 0.0, 0.0],
            [-1.2297560733739028, 2.13, 0.0],
            [0.0, 0.0, 10.0]
        )),
        scaled_positions=np.array((
            [0.3333333333333333, 0.6666666666666666, 0.5],
            [0.6666666666666667, 0.33333333333333337, 0.5]
        )),
    )
    sys2d = sys2d.repeat((3, 3, 1))

    # 3D system
    sys3d = ase.lattice.cubic.Diamond(
        size=(1, 1, 1),
        symbol='Si',
        pbc=True,
        latticeconstant=5.430710)

    def test(self):
        # Test with two radii setting alternatives
        for radii, cluster_threshold in [
                ("covalent", 2.7),
                ("vdw_covalent", 0.1),
            ]:

            # 0d_n_pbc0
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys0d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 0)

            # 0d_n_pbc3
            DimensionalityTests.sys1d.set_pbc([True, True, True])
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys0d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 0)

            # 1d_n_pbc3
            DimensionalityTests.sys1d.set_pbc([True, True, True])
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys1d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 1)

            # 1d_n_pbc2
            DimensionalityTests.sys1d.set_pbc([False, True, True])
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys1d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 1)

            # 1d_n_pbc1
            DimensionalityTests.sys1d.set_pbc([False, False, True])
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys1d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 1)

            # 2d_n_pbc3
            DimensionalityTests.sys2d.set_pbc([True, True, True])
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys2d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 2)

            # 2d_n_pbc2
            DimensionalityTests.sys2d.set_pbc([True, True, False])
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys2d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 2)

            # 3d_n_pbc3
            dimensionality = matid.geometry.get_dimensionality(
                DimensionalityTests.sys3d,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 3)

            # Test a system that has a non-orthogonal cell.
            system = ase.io.read("./structures/ROJiORHNwL4q0WTvNUy0mW5s2Buuq+PSX9X4dQR2r1cjQ9kBtuC-wI6MO8B.xyz")
            dimensionality = matid.geometry.get_dimensionality(
                system,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 3)

            # Test a surface that has been split by the cell boundary
            system = bcc100('Fe', size=(5, 1, 3), vacuum=8)
            system.translate([0, 0, 9])
            system.set_pbc(True)
            system.wrap(pbc=True)
            dimensionality = matid.geometry.get_dimensionality(
                system,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 2)

            # Test a surface with a high amplitude wave. This would break a
            # regular linear vacuum gap search.
            system = bcc100('Fe', size=(15, 3, 3), vacuum=8)
            pos = system.get_positions()
            x_len = np.linalg.norm(system.get_cell()[0, :])
            x = pos[:, 0]
            z = pos[:, 2]
            z_new = z + 3*np.sin(4*(x/x_len)*np.pi)
            pos_new = np.array(pos)
            pos_new[:, 2] = z_new
            system.set_positions(pos_new)
            system.set_pbc(True)
            dimensionality = matid.geometry.get_dimensionality(
                system,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 2)

            # Graphite
            system = ase.lattice.hexagonal.Graphite(
                size=(1, 1, 1),
                symbol='C',
                pbc=True,
                latticeconstant=(2.461, 6.708))
            dimensionality = matid.geometry.get_dimensionality(
                system,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 3)

            # Molecule with quite little vacuum between cells
            system = Atoms(
                symbols=['C', 'O'],
                positions=[
                    [1.13250529, 0., 0.],
                    [0., 0., 0.],
                ],
                cell=[
                    [5.29177211, 0., 0.],
                    [0., 5.29177211, 0.],
                    [0., 0., 5.29177211],
                ],
                pbc=True
            )
            dimensionality = matid.geometry.get_dimensionality(
                system,
                radii=radii,
                cluster_threshold=cluster_threshold)
            self.assertEqual(dimensionality, 0)


class PeriodicFinderTests(unittest.TestCase):
    """Unit tests for the class that is used to find periodic regions.
    """
    classifier = Classifier()
    max_cell_size = classifier.max_cell_size
    angle_tol = classifier.angle_tol
    delaunay_threshold = classifier.delaunay_threshold
    bond_threshold = classifier.bond_threshold
    pos_tol = classifier.pos_tol
    pos_tol_scaling = classifier.pos_tol_scaling
    cell_size_tol = classifier.cell_size_tol

    def test_cell_selection(self):
        """Testing that the correct cell is selected.
        """
        # 3D: Selecting orthogonal from two options with same volume
        spans = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 1],
        ])
        metrics = np.array([0, 0, 0, 0])

        finder = PeriodicFinder()
        indices = finder._find_best_basis(spans, metrics)
        self.assertTrue(np.array_equal(indices, np.array([0, 1, 2])))

        # 3D: Selecting the non-orthogonal because another combination has higer
        # periodicity
        spans = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 1],
        ])
        metrics = np.array([2, 2, 1, 2])

        finder = PeriodicFinder()
        indices = finder._find_best_basis(spans, metrics)
        self.assertTrue(np.array_equal(indices, np.array([0, 1, 3])))

        # 3D: Selecting first by volume, then by orthogonality.
        spans = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0.5, 0.5],
        ])
        metrics = np.array([0, 0, 0, 0])

        finder = PeriodicFinder()
        indices = finder._find_best_basis(spans, metrics)
        self.assertTrue(np.array_equal(indices, np.array([0, 1, 3])))

        # 2D: Selecting orthogonal from two options with same volume
        spans = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ])
        metrics = np.array([0, 0, 0])

        finder = PeriodicFinder()
        indices = finder._find_best_basis(spans, metrics)
        self.assertTrue(np.array_equal(indices, np.array([0, 1])))

        # 2D: Selecting the non-orthogonal because another combination has higer
        # periodicity
        spans = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 2, 0],
        ])
        metrics = np.array([2, 1, 2])

        finder = PeriodicFinder()
        indices = finder._find_best_basis(spans, metrics)
        self.assertTrue(np.array_equal(indices, np.array([0, 2])))

        # 2D: Selecting first by area, then by orthogonality.
        spans = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0.5, 0],
        ])
        metrics = np.array([0, 0, 0])

        finder = PeriodicFinder()
        indices = finder._find_best_basis(spans, metrics)
        self.assertTrue(np.array_equal(indices, np.array([0, 2])))

    def test_random(self):
        """Test a structure with random atom positions. No structure should be
        found.
        """
        n_atoms = 50
        rng = RandomState(8)
        for i in range(10):
            rand_pos = rng.rand(n_atoms, 3)

            system = Atoms(
                scaled_positions=rand_pos,
                cell=(10, 10, 10),
                symbols=n_atoms*['C'],
                pbc=(1, 1, 1))
            # view(system)

            finder = PeriodicFinder()
            region = finder.get_region(
                system,
                0,
                pos_tol=1,
                max_cell_size=8,
            )
            self.assertEqual(region, None)

    def test_nanocluster(self):
        """Test the periodicity finder on an artificial perfect nanocluster.
        """
        system = bcc100('Fe', size=(7, 7, 12), vacuum=0)
        system.set_cell([30, 30, 30])
        system.set_pbc(True)
        system.center()

        # Make the thing spherical
        center = np.array([15, 15, 15])
        pos = system.get_positions()
        dist = np.linalg.norm(pos - center, axis=1)
        valid_ind = dist < 10
        system = system[valid_ind]

        # Get the index of the atom that is closest to center of mass
        cm = system.get_center_of_mass()
        seed_index = np.argmin(np.linalg.norm(pos-cm, axis=1))
        # view(system)

        # Find the region with periodicity
        finder = PeriodicFinder()
        region = finder.get_region(
            system,
            seed_index,
            pos_tol=0.01,
            max_cell_size=4,
        )

        # view(region.cell)

        # No defects or unknown atoms
        adsorbates = region.get_adsorbates()
        interstitials = region.get_interstitials()
        substitutions = region.get_substitutions()
        vacancies = region.get_vacancies()
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)

    # def test_optimized_nanocluster(self):
        # """Test the periodicity finder on a DFT-optimized nanocluster. This
        # test does not yet pass because the full cluster is not detected
        # correctly.
        # """
        # system = ase.io.read("./structures/cu55.xyz")
        # system.set_cell([20, 20, 20])
        # system.set_pbc(True)
        # system.center()

        # # Get the index of the atom that is closest to center of mass
        # cm = system.get_center_of_mass()
        # pos = system.get_positions()
        # seed_index = np.argmin(np.linalg.norm(pos-cm, axis=1))
        # view(system)

        # # Find the region with periodicity
        # finder = PeriodicFinder()
        # region = finder.get_region(system, seed_index, 4, 2.75)
        # # print(region)

        # rec = region.recreate_valid()
        # view(rec)
        # # view(rec.unit_cell)

        # # No defects or unknown atoms
        # adsorbates = region.get_adsorbates()
        # interstitials = region.get_interstitials()
        # substitutions = region.get_substitutions()
        # vacancies = region.get_vacancies()
        # unknowns = region.get_unknowns()
        # self.assertEqual(len(interstitials), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)


class DelaunayTests(unittest.TestCase):
    """Tests for the Delaunay triangulation.
    """
    classifier = Classifier()
    delaunay_threshold = classifier.delaunay_threshold

    def test_surface(self):
        system = bcc100('Fe', size=(5, 5, 3), vacuum=8)
        # view(system)
        decomposition = matid.geometry.get_tetrahedra_decomposition(
            system,
            DelaunayTests.delaunay_threshold
        )

        # Atom inside
        test_pos = np.array([7, 7, 9.435])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)

        # Atoms at the edges should belong to the surface
        test_pos = np.array([14, 2, 9.435])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)
        test_pos = np.array([1.435, 13, 9.435])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)

        # Atoms outside
        test_pos = np.array([5, 5, 10.9])
        self.assertEqual(decomposition.find_simplex(test_pos), None)
        test_pos = np.array([5, 5, 7.9])
        self.assertEqual(decomposition.find_simplex(test_pos), None)

    def test_2d(self):
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(2, 2, 1),
            vacuum=8)
        system.set_pbc(True)
        # view(system)

        decomposition = matid.geometry.get_tetrahedra_decomposition(
            system,
            DelaunayTests.delaunay_threshold
        )

        # Atom inside
        test_pos = np.array([2, 2, 10])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)
        test_pos = np.array([2, 2, 10.5])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)

        # # Atoms at the edges should belong to the surface
        test_pos = np.array([0, 4, 10])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)
        test_pos = np.array([5, 1, 10])
        self.assertNotEqual(decomposition.find_simplex(test_pos), None)

        # # Atoms outside
        test_pos = np.array([2, 2, 11.2])
        self.assertEqual(decomposition.find_simplex(test_pos), None)
        test_pos = np.array([0, 0, 7.9])
        self.assertEqual(decomposition.find_simplex(test_pos), None)


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


class Class0DTests(unittest.TestCase):
    """Tests for detecting zero-dimensional systems.
    """
    def test_h2o_no_pbc(self):
        h2o = molecule("H2O")
        classifier = Classifier()
        clas = classifier.classify(h2o)
        self.assertIsInstance(clas, Class0D)

    def test_h2o_pbc(self):
        h2o = molecule("CH4")
        gap = 10
        h2o.set_cell([[gap, 0, 0], [0, gap, 0], [0, 0, gap]])
        h2o.set_pbc([True, True, True])
        h2o.center()
        classifier = Classifier()
        clas = classifier.classify(h2o)
        self.assertIsInstance(clas, Class0D)

    def test_unknown_molecule(self):
        """An unknown molecule should be classified as Class0D
        """
        system = Atoms(
            positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            symbols=["Au", "Ag"]
        )
        gap = 12
        system.set_cell([[gap, 0, 0], [0, gap, 0], [0, 0, gap]])
        system.set_pbc([True, True, True])
        system.center()
        # view(system)
        classifier = Classifier()
        clas = classifier.classify(system)
        self.assertIsInstance(clas, Class0D)


class Class1DTests(unittest.TestCase):
    """Tests detection of one-dimensional structures.
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
        self.assertIsInstance(clas, Class1D)

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
        self.assertIsInstance(clas, Class1D)

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
        self.assertIsInstance(clas, Class1D)

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
        self.assertIsInstance(clas, Class1D)


class Material2DTests(unittest.TestCase):
    """Tests detection of 2D structures.
    """
    graphene = Atoms(
        symbols=[6, 6],
        cell=np.array((
            [2.4595121467478055, 0.0, 0.0],
            [-1.2297560733739028, 2.13, 0.0],
            [0.0, 0.0, 20.0]
        )),
        scaled_positions=np.array((
            [0.3333333333333333, 0.6666666666666666, 0.5],
            [0.6666666666666667, 0.33333333333333337, 0.5]
        )),
        pbc=True
    )

    def test_molecule_network(self):
        """Test that a molecule network is not classified as a 2D material
        because of too sparse cell.
        """
        system = ase.io.read("./structures/R6JuJXj20goPQ0vv6aAVYpNyuwGgN+P_PaYo5EiiPChgUe9B6JnTX6BcOwt.xyz")
        # view(system)

        classifier = Classifier(max_cell_size=20, max_2d_cell_height=20)
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class2D)

    def test_small_cell_defect(self):
        """Test for a system with a defect and a simulation cell that is
        smaller than the maximum cell size. Currently such systems are labeled
        as Class2D as long as the simulation cell sizes are smaller than
        \l_{max}^{2D}.
        """
        system = Material2DTests.graphene.repeat([3, 3, 1])
        del system[8]
        system.set_pbc([True, True, False])
        # view(system)

        classifier = Classifier(max_cell_size=20)
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class2D)

    def test_small_cell_adsorption(self):
        """Test for a system with a defect and a simulation cell that is
        smaller than the maximum cell size. Currently such systems are labeled
        as Class2D as long as the simulation cell sizes are smaller than
        \l_{max}^{2D}.
        """
        system = Material2DTests.graphene.repeat([3, 3, 1])
        system.set_pbc([True, True, True])
        adsorbate = ase.Atom(position=[2, 2, 11], symbol="H")
        system += adsorbate
        # view(system)

        classifier = Classifier(max_cell_size=20)
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class2D)

    def test_small_2d_cell_vacuum_direction_included(self):
        """Test that the classification can properly handle systems where
        initially three basis vectors are detected, they are reduced to two due
        to wrong dimensionality of the cell, and then although only one
        repetition of the cell is found, it is accepted because its size is
        below the threshold MAX_SINGLE_CELL_SIZE.
        """
        system = ase.io.read("./structures/RJv-r5Vwf6ypWElBSq_hTCOaxEU89+PgZTqAjcn_4hHS3fozZkAI0Jxtdas.xyz")
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_vacuum_in_2d_unit_cell(self):
        """Structure where a 2D unit cell is found, but it has a vacuum gap.
        Should be detected by using TSA on the cell.
        """
        system = ase.io.read("./structures/RloVGNkMhI83gtwzF5DmftT6fM31d+P9ZCykgTQkZ7aIFmr-vje9gq8p6fc.xyz")

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Class2D)

    def test_graphene_sheets_close(self):
        """2D materials with a relatively small vacuum gap should be correctly
        identified. If a proper check is not done on the connectivity of the
        unit cell, these kind of structures may get classified as surfaces if
        the maximum cell size is bigger than the vacuum gap.
        """
        system = Material2DTests.graphene.repeat([3, 3, 1])
        old_cell = system.get_cell()
        old_cell[2, 2] = 10
        system.set_cell(old_cell)
        system.center()
        system.set_pbc([True, True, True])
        # view(system)

        classifier = Classifier(max_cell_size=12)
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Material2D)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_too_big_single_cell(self):
        """Test that with when only the simulation cell itself is the found
        unit cell, but the cell size is above a threshold value, the system
        cannot be classified.
        """
        system = Material2DTests.graphene.repeat([3, 3, 1])
        system.set_pbc([True, True, True])

        rng = RandomState(8)
        matid.geometry.make_random_displacement(system, 2, rng)

        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class2D)

    def test_seed_not_in_cell(self):
        """In this case the seed atom is not in the cells when creating
        the prototype cell. If the seed atom is directly at the cell border, it
        might not be found. This tests that the code forces the seed atom to be
        found correctly.
        """
        system = ase.io.read("./structures/RJv-r5Vwf6ypWElBSq_hTCOaxEU89+PKPif9Fqbl30oVX-710UwCHGMd83y.xyz")

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

    def test_layered_2d(self):
        """A stacked two-dimensional material. One of the materials should be
        recognized and the other recognized as adsorbate.
        """
        system = ase.io.read("./structures/RJv-r5Vwf6ypWElBSq_hTCOaxEU89+PDLFIM7Xvy9JaEqwS72kDtDr_Szhp.xyz")

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Material2D)

        # Boron nitrate adsorbate
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 4)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(set(adsorbates), set([8, 9, 10, 11]))

    def test_graphene_primitive(self):
        system = Material2DTests.graphene
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_graphene_rectangular(self):
        system = Atoms(
            symbols=["C", "C", "C", "C"],
            cell=np.array((
                [4.26, 0.0, 0.0],
                [0.0, 15, 0.0],
                [0.0, 0.0, 2.4595121467478055]
            )),
            positions=np.array((
                [2.84, 7.5, 6.148780366869514e-1],
                [3.55, 7.5, 1.8446341100608543],
                [7.1e-1, 7.5, 1.8446341100608543],
                [1.42, 7.5, 6.148780366869514e-1],
            )),
            pbc=True
        )
        system = system.repeat([2, 1, 2])
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # Pristine
        basis = classification.basis_indices
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(set(basis), set(range(len(system))))

    def test_2d_z_smaller_than_rmax(self):
        """Test that 2D systems that have an interlayer spacing smaller than
        r_max, the distance between different layers is not considered as a
        valid cell basis vector.
        """
        r_max = 12
        system = Atoms(
            symbols=["C", "C", "C", "C"],
            cell=np.array((
                [4.26, 0.0, 0.0],
                [0.0, 0.75*r_max, 0.0],
                [0.0, 0.0, 2.4595121467478055]
            )),
            positions=np.array((
                [2.84, 7.5, 6.148780366869514e-1],
                [3.55, 7.5, 1.8446341100608543],
                [7.1e-1, 7.5, 1.8446341100608543],
                [1.42, 7.5, 6.148780366869514e-1],
            )),
            pbc=True
        )
        system.center()
        system = system.repeat([2, 1, 2])
        # view(system)

        classifier = Classifier(max_cell_size=r_max)
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

    def test_graphene_supercell(self):
        system = Material2DTests.graphene.repeat([5, 5, 1])
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_graphene_partial_pbc(self):
        system = Material2DTests.graphene.copy()
        system.set_pbc([True, True, False])
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_graphene_missing_atom(self):
        """Test graphene with a vacancy defect.
        """
        system = Material2DTests.graphene.repeat([5, 5, 1])
        del system[24]
        # view(system)
        system.set_pbc([True, True, False])
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # One vacancy
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 1)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_graphene_substitution(self):
        """Test graphene with a substitution defect.
        """
        system = Material2DTests.graphene.repeat([5, 5, 1])
        system[0].number = 7
        # view(system)
        system.set_pbc([True, True, False])
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # One substitution
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 1)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

        # Check substitution info
        subst = substitutions[0]
        index = subst.index
        orig_num = subst.original_element
        subst_num = subst.substitutional_element
        self.assertEqual(index, 0)
        self.assertEqual(orig_num, 6)
        self.assertEqual(subst_num, 7)

    def test_graphene_missing_atom_exciting(self):
        """Test a more realistic graphene with a vacancy defect from the
        exciting data in the NOMAD Archive.
        """
        positions = np.array([[0.0, 0.0, 0.0],
            [0.0, 9.833294145128265E-10, 0.0],
            [2.134121238221869E-10, -1.23213547309968E-10, 0.0],
            [2.8283321482383327E-10, 9.83786883934224E-10, 0.0],
            [7.159944277047908E-11, 1.2149852888233143E-10, 0.0],
            [9.239798421116619E-10, 3.6970883192833546E-10, 0.0],
            [7.159944277047908E-11, 8.618308856304952E-10, 0.0],
            [9.239798421116619E-10, 6.136207055601422E-10, 0.0],
            [2.8283321482383327E-10, -4.573464457464822E-13, 0.0],
            [4.2635394347838356E-10, -2.458942411245288E-10, 0.0],
            [1.0647740633039121E-9, -3.6912488204997373E-10, 0.0],
            [8.52284868807466E-10, 2.4537848124459853E-10, 0.0],
            [1.0647740633039121E-9, 1.2269778743003765E-10, 0.0],
            [8.52284868807466E-10, -4.918055758645343E-10, 0.0],
            [4.2635394347838356E-10, -5.328534954072828E-13, 0.0],
            [4.970111804163183E-10, 8.604516522176773E-10, 0.0],
            [7.132179717248617E-11, 3.686497656226703E-10, 0.0],
            [7.100794156171322E-10, 2.4589288839236865E-10, 0.0],
            [7.132179717248617E-11, 6.146797718658073E-10, 0.0],
            [7.100794156171322E-10, 7.374366490961087E-10, 0.0],
            [4.970111804163183E-10, 1.2287788527080025E-10, 0.0],
            [6.39163064087745E-10, 8.6063580825492E-10, 0.0],
            [8.637153048417516E-14, 4.916647072564134E-10, 0.0],
            [6.39163064087745E-10, 1.2269360625790666E-10, 0.0],
            [2.1331073578640276E-10, 1.2303793808046385E-10, 0.0],
            [8.517910281331687E-10, 4.916647072564134E-10, 0.0],
            [2.1331073578640276E-10, 8.602914764323629E-10, 0.0],
            [4.970778494398485E-10, -1.232134858221425E-10, 0.0],
            [9.231674598249378E-10, -3.6921643742207865E-10, 0.0],
            [9.231675663249753E-10, 1.227894042899681E-10, 0.0],
            [2.84056580755611E-10, 2.4557345913912146E-10, 0.0],
            [7.102992316947146E-10, 4.916647687442388E-10, 0.0],
            [2.84056580755611E-10, 7.377560783493561E-10, 0.0],
            [6.391754180921053E-10, -1.2321354730996796E-10, 0.0],
            [8.521187287488282E-10, -2.461564252122759E-10, 0.0],
            [8.521187287488282E-10, -2.706694076601711E-13, 0.0],
            [7.101400141385201E-10, -2.4618501705111326E-10, 0.0],
            [9.231328473127216E-10, -1.23213547309968E-10, 0.0],
            [7.101400141385201E-10, -2.4207756882281025E-13, 0.0],
            [2.84140396285193E-10, 4.916647687442387E-10, 0.0],
            [4.971359984603718E-10, 3.6869170031963166E-10, 0.0],
            [4.971361049604094E-10, 6.146377756810205E-10, 0.0],
            [2.1311743821817984E-10, 3.6878393205781663E-10, 0.0],
            [6.390654035532765E-10, 3.6862443263858213E-10, 0.0],
            [4.262295514344803E-10, 7.375859415363175E-10, 0.0],
            [6.390654035532765E-10, 6.147051048498954E-10, 0.0],
            [4.262295514344803E-10, 2.4574359595216E-10, 0.0],
            [2.1311743821817984E-10, 6.145456054306609E-10, 0.0],
            [4.2613753540200396E-10, 4.916647687442388E-10, 0.0]
        ])
        labels = ["C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"]
        cell = np.array([
            [1.0650003758837873E-9, -6.148782545663813E-10, 0.0],
            [0.0, 1.2297565091327626E-9, 0.0],
            [0.0, 0.0, 2.0000003945832858E-9]
        ])
        pbc = True

        system = ase.Atoms(
            positions=1e10*positions,
            symbols=labels,
            cell=1e10*cell,
            pbc=pbc,
        )
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)
        # view(classification.region.recreate_valid())

        # One vacancy
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 1)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

        # Check vacancy position
        vac_atom = vacancies[0]
        vac_symbol = vac_atom.symbol
        vac_pos = vac_atom.position
        self.assertEqual(vac_symbol, "C")
        self.assertTrue(np.allclose(vac_pos, [0.7123, 11.0639, 0], atol=0.05))

    def test_graphene_shaken(self):
        """Test graphene that has randomly oriented but uniform length
        dislocations.
        """
        # Run multiple times with random displacements
        rng = RandomState(7)
        for i in range(15):
            system = Material2DTests.graphene.repeat([5, 5, 1])
            matid.geometry.make_random_displacement(system, 0.1, rng)
            classifier = Classifier()
            classification = classifier.classify(system)
            self.assertIsInstance(classification, Material2D)

            # Pristine
            adsorbates = classification.adsorbates
            interstitials = classification.interstitials
            substitutions = classification.substitutions
            vacancies = classification.vacancies
            unknowns = classification.unknowns
            if len(vacancies) != 0:
                view(system)
                view(classification.region.cell)
            self.assertEqual(len(interstitials), 0)
            self.assertEqual(len(substitutions), 0)
            self.assertEqual(len(vacancies), 0)
            self.assertEqual(len(adsorbates), 0)
            self.assertEqual(len(unknowns), 0)

    def test_chemisorption(self):
        """Test the adsorption where there is sufficient distance between the
        adsorbate and the surface to distinguish between them even if they
        share the same elements.
        """
        system = ase.io.read("./structures/RloVGNkMhI83gtwzF5DmftT6fM31d+PKxGoPkNrvdpZrlLS-V14MszJ-57L.xyz")
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # No defects or unknown atoms, one adsorbate cluster
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(adsorbates), 24)
        self.assertTrue(np.array_equal(adsorbates, np.arange(50, 74)))

    def test_curved_2d(self):
        """Curved 2D-material
        """
        graphene = create_graphene()
        graphene = graphene.repeat([5, 5, 1])

        # Bulge the surface
        cell_width = np.linalg.norm(graphene.get_cell()[0, :])
        for atom in graphene:
            pos = atom.position
            distortion_z = 0.30*np.sin(pos[0]/cell_width*2.0*np.pi)
            pos += np.array((0, 0, distortion_z))

        classifier = Classifier()
        classification = classifier.classify(graphene)
        self.assertIsInstance(classification, Material2D)

    def test_mos2_pristine_supercell(self):
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(5, 5, 1),
            vacuum=8)
        system.set_pbc(True)
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # Pristine
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_mos2_pristine_primitive(self):
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(1, 1, 1),
            vacuum=8)
        system.set_pbc(True)
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # Pristine
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_mos2_substitution(self):
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(5, 5, 1),
            vacuum=8)
        system.set_pbc(True)

        symbols = system.get_atomic_numbers()
        symbols[25] = 6
        system.set_atomic_numbers(symbols)

        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # One substitution
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(substitutions), 1)

    def test_mos2_vacancy(self):
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(5, 5, 1),
            vacuum=8)
        system.set_pbc(True)

        del system[25]
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # One vacancy
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(vacancies), 1)

    def test_mos2_adsorption(self):
        """Test adsorption on mos2 surface.
        """
        system = ase.build.mx2(
            formula="MoS2",
            kind="2H",
            a=3.18,
            thickness=3.19,
            size=(5, 5, 1),
            vacuum=8)
        system.set_pbc(True)

        ads = molecule("C6H6")
        ads.translate([4.9, 5.5, 13])
        system += ads

        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # One adsorbate
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 12)
        self.assertTrue(np.array_equal(adsorbates, range(75, 87)))

    def test_2d_split(self):
        """A simple 2D system where the system has been split by the cell
        boundary.
        """
        system = Atoms(
            symbols=["H", "C"],
            cell=np.array((
                [2, 0.0, 0.0],
                [0.0, 2, 0.0],
                [0.0, 0.0, 15]
            )),
            positions=np.array((
                [0, 0, 0],
                [0, 0, 13.8],
            )),
            pbc=True
        )
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # Pristine
        basis = classification.basis_indices
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(set(basis), set(range(len(system))))

    def test_boron_nitride(self):
        system = Atoms(
            symbols=["B", "N"],
            cell=np.array((
                [2.412000008147063, 0.0, 0.0],
                [-1.2060000067194177, 2.0888532824002019, 0.0],
                [0.0, 0.0, 15.875316320100001]
            )),
            positions=np.array((
                [0, 0, 0],
                [-1.3823924100453746E-9, 1.3925688618963122, 0.0]
            )),
            pbc=True
        )
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # Pristine
        basis = classification.basis_indices
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(set(basis), set(range(len(system))))

    def test_fluorographene(self):
        system = Atoms(
            scaled_positions=np.array([
                [1.3012393333576103e-06, 0.9999449352451434, 0.07686917114285712],
                [0.666645333381887, 0.9999840320410395, 0.10381504828571426],
                [0.16664461721471663, 0.49999686527625936, 0.10381366714285713],
                [0.5000035589866841, 0.49995279413001426, 0.07686989028571428],
                [0.9999651360110703, 6.476326633588427e-05, 0.0026979231428571424],
                [0.6665936880181591, 6.312126818602304e-05, 0.17797979399999994],
                [0.16658826335530388, 0.5001281031872844, 0.1779785431428571],
                [0.49997811077528137, 0.5001300794718694, 0.002698536571428571]
            ]),
            cell=np.array([
                [4.359520614662661, 0.0, 0.0],
                [0.0, 2.516978484830788, 0.0],
                [0.0, 0.0, 18.521202373450003]
            ]),
            symbols=[6, 6, 6, 6, 9, 9, 9, 9],
            pbc=True
        )

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Material2D)

        # Pristine
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)


class Class3DTests(unittest.TestCase):
    """Tests detection of bulk 3D materials.
    """
    def test_thin_sparse(self):
        """Test a crystal that is very thin.
        """
        system = Atoms(
            scaled_positions=np.array([
                [0.875000071090061, 0.6250000710900608, 0.2499998578198783],
                [0.12499992890993901, 0.37499992890993905, 0.750000142180122],
                [0.624999928909939, 0.8749999289099393, 0.750000142180122],
                [0.37500007109006084, 0.12500007109006087, 0.2499998578198783]
            ]),
            symbols=[5, 5, 51, 51],
            cell=np.array([
                [10.1, 0.0, 0.0],
                [0.0, 10.1, 0.0],
                [5.05, 5.05, 1.758333]
            ]),
            pbc=[True, True, True],
        )

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class3D)

    def test_si(self):
        si = ase.lattice.cubic.Diamond(
            size=(1, 1, 1),
            symbol='Si',
            pbc=(1, 1, 1),
            latticeconstant=5.430710)
        classifier = Classifier()
        classification = classifier.classify(si)
        self.assertIsInstance(classification, Class3D)

    def test_si_shaken(self):
        rng = RandomState(47)
        for i in range(10):
            si = ase.lattice.cubic.Diamond(
                size=(1, 1, 1),
                symbol='Si',
                pbc=(1, 1, 1),
                latticeconstant=5.430710)
            matid.geometry.make_random_displacement(si, 0.2, rng)
            classifier = Classifier()
            classification = classifier.classify(si)
            self.assertIsInstance(classification, Class3D)

    def test_graphite(self):
        """Testing a sparse material like graphite.
        """
        system = ase.lattice.hexagonal.Graphite(
            size=(1, 1, 1),
            symbol='C',
            pbc=(1, 1, 1),
            latticeconstant=(2.461, 6.708))
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class3D)

    def test_amorphous(self):
        """Test an amorphous crystal with completely random positions. This is
        currently not classified as crystal, but the threshold can be set in
        the classifier setup.
        """
        n_atoms = 50
        rng = RandomState(8)
        rand_pos = rng.rand(n_atoms, 3)

        system = Atoms(
            scaled_positions=rand_pos,
            cell=(10, 10, 10),
            symbols=n_atoms*['C'],
            pbc=(1, 1, 1))
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Class3D)

    def test_too_sparse(self):
        """Test a crystal that is too sparse.
        """
        system = ase.lattice.hexagonal.Graphite(
            size=(1, 1, 1),
            symbol='C',
            pbc=(1, 1, 1),
            latticeconstant=(2.461, 12))

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Unknown)


class SurfaceTests(unittest.TestCase):
    """Tests for detecting and analyzing surfaces.
    """
    def test_adsorbate_pattern(self):
        """Here the adsorbate will easily get included in the basis if the
        values for \omega_v and \omega_c are not suitable.
        """
        system = ase.io.read("./structures/RmlNIfj-YIQ14UBYjtAHtXcAEXZif+PIkKcrxeOf997qnQ_hWRXLdMsmpAf.xyz")
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Surface)
        self.assertTrue(np.array_equal(classification.outliers, [24, 25, 26]))

    def test_not_enough_repetitions(self):
        """In this system there is not enough repetitions of the cell in a
        third direction. One can with visual inspection guess the cell, but the
        algorithm cannot find it.
        """
        system = ase.io.read("./structures/Rhn-EWQQN8Z-lbmZwoWPyrGiM9Isx+PbYDgCBSwbq3nxONqWaq03HYUn8_V.xyz")
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Class2D)

    def test_incorrect_cell(self):
        """The cell detected for this system is not the correct one. The cell
        detection fails because there is only two cells from which to extract
        information, and one of them is missing an atom.
        """
        system = ase.io.read("./structures/Rq0LUBXa6rZ-mddbQUZJXOIVAIg-J+Pm73-Kx5CWtuIHzLTr5R-Nir2el0i.xyz")
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Surface)
        # print(classification.outliers)

    def test_thin_surface(self):
        """A realistic surface with only two layers.
        """
        system = ase.io.read("./structures/RmlNIfj-YIQ14UBYjtAHtXcAEXZif+PYu3zrqdlNhhs9tII2lnvJ3Gj7tai.xyz")
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        # cell = classification.region.cell
        self.assertEqual(type(classification), Surface)
        self.assertTrue(np.array_equal(classification.outliers, [24, 25, 26]))

    def test_mo_incorrect_3(self):
        """System where the outlier detection fails currently. The carbon in a
        carbon dioxide adsorbate is very hard to distinguish from the surface.
        """
        system = ase.io.read("./structures/RmlNIfj-YIQ14UBYjtAHtXcAEXZif+PmZsb-Uf3AIGQyTBZDg4ZgxXaq5UB.xyz")
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Surface)
        # self.assertTrue(np.array_equal(classification.outliers, [24, 25, 26]))

    def test_2d_motif_in_surface_hard(self):
        """Test that if a 2D substructure is found within a surface, and the 2D
        substructure covers a lot of the structure, the entire structure is
        still not classified according to that motif if it does not wrap to
        itself in the two basis vector directions.
        """
        translation = np.array([0, 0, 2])
        n_rep = 3
        graphene = Material2DTests.graphene.repeat((n_rep, n_rep, 1))
        layer1 = graphene

        layer2 = graphene.copy()
        layer2.set_chemical_symbols(["O"]*len(layer2))
        rng = RandomState(47)
        matid.geometry.make_random_displacement(layer2, 1, rng)
        layer2.translate(translation)

        system = layer1 + layer2

        old_cell = system.get_cell()
        old_cell[0, :] *= 2
        old_cell[2, :] = np.array([0, 0, 4])
        system.set_cell(old_cell)
        system.center()
        # view(system)

        # Should be classified as Class2D because the 2D motif that is detected
        # is not really periodic in the found basis vector directions (does not
        # wrap to itself).
        classifier = Classifier(max_cell_size=3, pos_tol=0.25)
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Class2D)

    def test_2d_motif_in_surface_easy(self):
        """Test that if a 2D substructure is found within a surface, the entire
        structure is not classified according to that motif if it does not
        cover enough of the structure.
        """
        # Here we create a 2D system which has alternating layers of ordered
        # and disordered sheets of 2D materials, but rotated 90 degree with
        # respect to the surface plane.
        translation = np.array([0, 0, 2])
        n_rep = 3
        graphene = Material2DTests.graphene.repeat((n_rep, n_rep, 1))
        layer1 = graphene

        layer2 = graphene.copy()
        layer2.set_chemical_symbols(["O"]*len(layer2))
        rng = RandomState(47)
        matid.geometry.make_random_displacement(layer2, 1, rng)
        layer2.translate(translation)

        layer3 = layer1.copy()
        layer3.translate(2*translation)

        layer4 = graphene.copy()
        layer4.set_chemical_symbols(["N"]*len(layer2))
        rng = RandomState(47)
        matid.geometry.make_random_displacement(layer4, 1, rng)
        layer4.translate(3*translation)

        system = layer1 + layer2 + layer3 + layer4

        old_cell = system.get_cell()
        old_cell[0, :] *= 3
        old_cell[2, :] = np.array([0, 0, 8])
        system.set_cell(old_cell)
        system.center()
        # view(system)

        # Should be classified as Class2D because the coverage is too small
        classifier = Classifier(max_cell_size=4)
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Class2D)

    def test_surface_difficult_basis_atoms(self):
        """This is a surface where the atoms on top of the surface will get
        easily classified as adsorbates if the chemical environment detection
        is not tuned correctly.
        """
        system = ase.io.read("./structures/RzQh5XijWuXsNZiRSxeOlPFUY_9Gl+PY5NRLMRYyQXsYmBN9hMcT-FftquP.xyz")
        # view(system)

        # With a little higher chemical similarity threshold the whole surface
        # is not detected
        classifier = Classifier(chem_similarity_threshold=0.45)
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Has outliers with these settings
        outliers = classification.outliers
        self.assertTrue(len(outliers) != 0)

        # With a little lower chemical similarity threshold the whole surface
        # is again detected
        classifier = Classifier(chem_similarity_threshold=0.40)
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Has no outliers with these settings
        outliers = classification.outliers
        self.assertTrue(len(outliers) == 0)

    # def test_surface_with_one_cell_but_periodic_backbone(self):
        # """This is a surface that ultimately has only one repetition of the
        # underlying unit cell in the simulation cell. Normally it would not get
        # classified, but because it has a periodic backbone of Barium atoms,
        # they are identified as the unit cell and everything inside is
        # identified as outliers. Such systems still pose a challenge to the
        # algorithm.
        # """
        # system = ase.io.read("./structures/Rhn-EWQQN8Z-lbmZwoWPyrGiM9Isx+PbYDgCBSwbq3nxONqWaq03HYUn8_V.xyz")
        # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # No outliers
        # outliers = classification.outliers
        # self.assertEqual(len(outliers), 0)

    def test_adsorbate_detection_via_neighbourhood(self):
        """Test that adsorbates that are in a basis atom position, but do not
        exhibit the correct chemical neighbourhood are identified.
        """
        system = ase.io.read("./structures/ROHGEranIWm-gnS6jhQaLZRORWDKx+Pbsl6Hlb_C1aXadFiJ58UCUek5a8x.xyz")
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Only adsorbates
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 18)
        self.assertEqual(len(unknowns), 0)
        self.assertTrue(np.array_equal(adsorbates, np.arange(0, 18)))

    def test_surface_wrong_cm(self):
        """Test that the seed atom is correctly chosen near the center of mass
        even if the structure is cut.
        """
        system = bcc100('Fe', size=(3, 3, 4), vacuum=8)
        adsorbate = ase.Atom(position=[4, 4, 4], symbol="H")
        system += adsorbate
        system.set_pbc([True, True, True])
        system.translate([0, 0, 10])
        system.wrap()
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        # view(classification.region.recreate_valid())
        # view(classification.region.cell)
        self.assertIsInstance(classification, Surface)

        # One adsorbate
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 1)
        self.assertEqual(len(unknowns), 0)

    def test_search_beyond_limits(self):
        """In this system the found unit cell cannot be used to seach the whole
        surface unless seed atoms for unit cells beyond the original simulation
        cell boundaries are not allowed.
        """
        system = ase.io.read("./structures/RDtJ5cTyLBPt4PA182VbCzoCxf5Js+PEzXqLISX8Pam-HlJMxeLc86lcKgf.xyz")

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Only adsorbates
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(adsorbates), 14)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(interstitials), 0)

    def test_ordered_adsorbates(self):
        """Test surface where on top there are adsorbates with high
        connectivity in two directions. These kind of adsorbates could not be
        detected if the size of the connected components would not be checked.
        """
        system = ase.io.read("./structures/RDtJ5cTyLBPt4PA182VbCzoCxf5Js+P8Wnwz4dfyea6UAD0WEBadXv83wyf.xyz")
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Only adsorbates
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 13)
        self.assertEqual(len(unknowns), 0)
        self.assertTrue(np.array_equal(adsorbates, np.arange(0, 13)))

    def test_surface_with_one_basis_vector_as_span(self):
        system = ase.io.read("./structures/RDtJ5cTyLBPt4PA182VbCzoCxf5Js+PFw_-OtcPJ5og8XMItaAAFYhQUaY6.xyz")
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # view(classification.region.recreate_valid())

        # Only adsorbates
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 6)
        self.assertEqual(len(unknowns), 0)
        self.assertTrue(np.array_equal(adsorbates, np.arange(0, 6)))

    def test_cut_surface(self):
        """Test a surface that has been cut by the cell boundary. Should still
        be detected as single surface.
        """
        system = ase.io.read("./structures/RscdVKibS4pD0O_Yo1CSwkznfiL1c+PCvflj-qTkfRcUaCISfn8fm-2oaVW.xyz")
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Pristine
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_zinc_blende(self):
        system = Zincblende(symbol=["Au", "Fe"], latticeconstant=5)
        system = system.repeat((4, 4, 2))
        cell = system.get_cell()
        cell[2, :] *= 3
        system.set_cell(cell)
        system.center()
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # Check that the right cell is found
        proto_cell = classification.prototype_cell
        analyzer = SymmetryAnalyzer(proto_cell, symmetry_tol=0.4)
        space_group = analyzer.get_space_group_number()
        self.assertEqual(space_group, 216)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_thin_complex_surface(self):
        """Test for a complex thin surface with adsorbate. This surface has
        only two repetitions in the surface normal direction.
        """
        system = ase.io.read("./structures/RmlNIfj-YIQ14UBYjtAHtXcAEXZif+Pkl2CiGU9KP0uluTY8M3PeGEb4OS_.xyz")
        # view(system)

        classifier = Classifier(pos_tol=0.75)
        classification = classifier.classify(system)
        self.assertEqual(type(classification), Surface)

        # CO2 adsorbate
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(adsorbates), 3)
        self.assertTrue(np.array_equal(adsorbates, np.array([24, 25, 26])))

    def test_bcc_pristine_small_surface(self):
        system = bcc100('Fe', size=(1, 1, 3), vacuum=8)
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_bcc_pristine_big_surface(self):
        system = bcc100('Fe', size=(5, 5, 3), vacuum=8)
        # view(system)
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_bcc_substitution(self):
        """Surface with substitutional point defect.
        """
        system = bcc100('Fe', size=(5, 5, 3), vacuum=8)
        labels = system.get_atomic_numbers()
        sub_index = 42
        sub_element = 20
        labels[sub_index] = sub_element
        system.set_atomic_numbers(labels)
        # view(system)

        # Classified as surface
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # One substitutional defect
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(substitutions), 1)
        subst = substitutions[0]
        self.assertEqual(subst.index, sub_index)
        self.assertEqual(subst.original_element, 26)
        self.assertEqual(subst.substitutional_element, sub_element)

    def test_bcc_vacancy(self):
        """Surface with vacancy point defect.
        """
        system = bcc100('Fe', size=(5, 5, 3), vacuum=8)
        vac_index = 42

        # Get the vacancy atom
        vac_true = ase.Atom(
            system[vac_index].symbol,
            system[vac_index].position,
        )
        del system[vac_index]
        # view(system)

        # Classified as surface
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # One vacancy
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 1)
        vac_found = vacancies[0]
        self.assertTrue(np.allclose(vac_true.position, vac_found.position))
        self.assertEqual(vac_true.symbol, vac_found.symbol)

    def test_bcc_interstitional(self):
        """Surface with interstitional atom.
        """
        system = bcc100('Fe', size=(5, 5, 5), vacuum=8)

        # Add an interstitionl atom
        interstitional = ase.Atom(
            "C",
            [8, 8, 9],
        )
        system += interstitional

        # view(system)

        # Classified as surface
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # One interstitional
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(interstitials), 1)
        int_found = interstitials[0]
        self.assertEqual(int_found, 125)

    def test_bcc_dislocated_big_surface(self):
        system = bcc100('Fe', size=(5, 5, 3), vacuum=8)

        # Run multiple times with random displacements
        rng = RandomState(47)
        # for i in range(4):
            # disloc = rng.rand(len(system), 3)
        for i in range(10):
            i_sys = system.copy()
            matid.geometry.make_random_displacement(system, 0.04, rng)
            # view(system)

            # Classified as surface
            # classifier = Classifier(pos_tol=0.75)
            classifier = Classifier()
            classification = classifier.classify(i_sys)
            self.assertIsInstance(classification, Surface)

            # No defects or unknown atoms
            adsorbates = classification.adsorbates
            interstitials = classification.interstitials
            substitutions = classification.substitutions
            vacancies = classification.vacancies
            unknowns = classification.unknowns
            self.assertEqual(len(interstitials), 0)
            self.assertEqual(len(substitutions), 0)
            self.assertEqual(len(vacancies), 0)
            self.assertEqual(len(adsorbates), 0)
            self.assertEqual(len(unknowns), 0)

    def test_curved_surface(self):
        # Create an Fe 100 surface as an ASE Atoms object
        system = bcc100('Fe', size=(12, 12, 3), vacuum=8)

        # Bulge the surface
        cell_width = np.linalg.norm(system.get_cell()[0, :])
        for atom in system:
            pos = atom.position
            distortion_z = 0.5*np.sin(pos[0]/cell_width*2.0*np.pi)
            pos += np.array((0, 0, distortion_z))
        # view(system)

        # Classified as surface
        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # No defects or unknown atoms
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns
        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(adsorbates), 0)
        self.assertEqual(len(unknowns), 0)

    def test_surface_ads(self):
        """Test a surface with an adsorbate.
        """
        # Create an Fe 100 surface as an ASE Atoms object
        system = bcc100('Fe', size=(5, 5, 4), vacuum=8)

        # Add a H2O molecule on top of the surface
        h2o = molecule("H2O")
        h2o.rotate(180, [1, 0, 0])
        h2o.translate([7.2, 7.2, 13.5])
        system += h2o
        # view(system)

        classifier = Classifier()
        classification = classifier.classify(system)
        self.assertIsInstance(classification, Surface)

        # No defects or unknown atoms, one adsorbate cluster
        adsorbates = classification.adsorbates
        interstitials = classification.interstitials
        substitutions = classification.substitutions
        vacancies = classification.vacancies
        unknowns = classification.unknowns

        self.assertEqual(len(interstitials), 0)
        self.assertEqual(len(substitutions), 0)
        self.assertEqual(len(vacancies), 0)
        self.assertEqual(len(unknowns), 0)
        self.assertEqual(len(adsorbates), 3)
        self.assertTrue(np.array_equal(adsorbates, np.array([100, 101, 102])))

    def test_nacl(self):
        """Test the detection for an imperfect NaCl surface with adsorbate and
        defects.
        """
        # Create the system
        class NaClFactory(SimpleCubicFactory):
            "A factory for creating NaCl (B1, Rocksalt) lattices."

            bravais_basis = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5],
                            [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0],
                            [0.5, 0.5, 0.5]]
            element_basis = (0, 1, 1, 0, 1, 0, 0, 1)

        nacl = NaClFactory()
        nacl = nacl(symbol=["Na", "Cl"], latticeconstant=5.64)
        nacl = nacl.repeat((4, 4, 2))
        cell = nacl.get_cell()
        cell[2, :] *= 3
        nacl.set_cell(cell)
        nacl.center()

        # Add vacancy
        vac_index = 17
        vac_true = ase.Atom(
            nacl[vac_index].symbol,
            nacl[vac_index].position,
        )
        del nacl[vac_index]

        # Shake the atoms
        # rng = RandomState(8)
        # matid.geometry.make_random_displacement(nacl, 0.4, rng)

        # Add adsorbate
        h2o = molecule("H2O")
        h2o.rotate(-45, [0, 0, 1])
        h2o.translate([11.5, 11.5, 22.5])
        nacl += h2o

        # Add substitution
        symbols = nacl.get_atomic_numbers()
        subst_num = 39
        subst_atomic_num = 19
        symbols[subst_num] = subst_atomic_num
        nacl.set_atomic_numbers(symbols)

        # view(nacl)

        classifier = Classifier()
        classification = classifier.classify(nacl)
        self.assertIsInstance(classification, Surface)

        # Detect adsorbate
        adsorbates = classification.adsorbates
        # print(adsorbates)
        self.assertEqual(len(adsorbates), 3)
        self.assertTrue(np.array_equal(adsorbates, np.array([256, 257, 255])))

        # Detect vacancy
        vacancies = classification.vacancies
        self.assertEqual(len(vacancies), 1)
        vac_found = vacancies[0]
        vacancy_disp = np.linalg.norm(vac_true.position - vac_found.position)
        self.assertTrue(vacancy_disp <= 1)
        self.assertEqual(vac_true.symbol, vac_found.symbol)

        # Detect substitution
        substitutions = classification.substitutions
        self.assertEqual(len(substitutions), 1)
        found_subst = substitutions[0]
        self.assertEqual(found_subst.index, subst_num)
        self.assertEqual(found_subst.original_element, 11)
        self.assertEqual(found_subst.substitutional_element, subst_atomic_num)

        # No unknown atoms
        unknowns = classification.unknowns
        self.assertEqual(len(unknowns), 0)

        # No interstitials
        interstitials = classification.interstitials
        self.assertEqual(len(interstitials), 0)


class SearchGraphTests(unittest.TestCase):
    """For testing the detection of finite regions by analyzing the
    connectivity of different unit cell.
    """
    def test_non_orthogonal_cell_1(self):
        """Non-orthogonal cell with only one atom.
        """
        cell = np.array([
            [7.8155, 0., 0.],
            [-3.9074, 6.7683, 0.],
            [0., 0., 175.]
        ])
        cell[0:2, :] *= 0.5
        pos = np.array([
            [0.0, 0.0, 0.5],
        ])
        symbols = np.array(["Sr"])
        system = Atoms(
            scaled_positions=pos,
            cell=cell,
            symbols=symbols,
            pbc=True
        )
        # view(system)

        finder = PeriodicFinder()
        region = finder.get_region(system, 0, 5, 0.7)
        # view(region.cell)

        G = region._search_graph
        # draw_networkx(G)
        # mpl.show()

        # Check that the correct graph is created
        self.assertEqual(len(G.nodes), 1)
        self.assertEqual(len(G.edges), 8)

        # Check graph periodicity
        periodicity = region.get_connected_directions()
        self.assertTrue(np.array_equal(periodicity, [True, True, False]))

    def test_non_orthogonal_cell_2(self):
        """Non-orthogonal cell with two atoms.
        """
        cell = np.array([
            [7.8155, 0., 0.],
            [-3.9074, 6.7683, 0.],
            [0., 0., 175.]
        ])
        cell[1:2, :] *= 0.5
        pos = np.array([
            [0.66041568, 0.64217915, 0.49500249],
            [0.1463031, 0.60235042, 0.49423654],
        ])
        symbols = np.array(2*["Sr"])
        system = Atoms(
            scaled_positions=pos,
            cell=cell,
            symbols=symbols,
            pbc=True
        )
        # view(system)

        finder = PeriodicFinder()
        region = finder.get_region(system, 0, 5, 0.8)
        # view(region.cell)

        G = region._search_graph
        # draw_networkx(G)
        # mpl.show()

        # Check that the correct graph is created
        self.assertEqual(len(G.nodes), 2)
        self.assertEqual(len(G.edges), 11)

        # Check graph periodicity
        periodicity = region.get_connected_directions()
        self.assertTrue(np.array_equal(periodicity, [True, True, False]))

    def test_non_orthogonal_cell_4(self):
        """Non-orthogonal cell with four atoms.
        """
        cell = np.array([
            [7.8155, 0., 0.],
            [-3.9074, 6.7683, 0.],
            [0., 0., 175.]
        ])
        pos = np.array([
            [0.66041568, 0.64217915, 0.49500249],
            [0.63081094, 0.13665159, 0.49460691],
            [0.1463031, 0.60235042, 0.49423654],
            [0.11211634, 0.1241777, 0.49450267]
        ])
        symbols = np.array(4*["Sr"])
        system = Atoms(
            scaled_positions=pos,
            cell=cell,
            symbols=symbols,
            pbc=True
        )
        # view(system)

        finder = PeriodicFinder()
        region = finder.get_region(system, 0, 5, 0.7)
        G = region._search_graph

        # Check that the correct graph is created
        self.assertEqual(len(G.nodes), 4)
        self.assertEqual(len(G.edges), 22)

        # Check graph periodicity
        periodicity = region.get_connected_directions()
        self.assertTrue(np.array_equal(periodicity, [True, True, False]))

    def test_surface_difficult_basis_atoms(self):
        """This system with this specific position tolerance fails if there is
        no check against moves that occur inside the unit cell 'grid', and do
        not wrap across it.
        """
        system = ase.io.read("./structures/RzQh5XijWuXsNZiRSxeOlPFUY_9Gl+PY5NRLMRYyQXsYmBN9hMcT-FftquP.xyz")
        # view(system)

        finder = PeriodicFinder()
        region = finder.get_region(system, 42, 12, 1.05146337551)

        # Check graph periodicity
        periodicity = region.get_connected_directions()
        self.assertTrue(np.array_equal(periodicity, [False, True, True]))

    def test_surface_adsorbate(self):
        """Test graph search in the presence of adsorbates.
        """
        system = ase.io.read("./structures/ROHGEranIWm-gnS6jhQaLZRORWDKx+Pbco91p05ftuJQ38__Y0_TDg9tNIy.xyz")
        # view(system)

        finder = PeriodicFinder()
        region = finder.get_region(system, 19, 12, 0.252687223066)
        G = region._search_graph

        # Check that the correct graph is created
        self.assertEqual(len(G.nodes), 16)
        # self.assertEqual(len(G.edges), 64) # All edges not found due to low pos_tol

        # Check graph periodicity
        periodicity = region.get_connected_directions()
        self.assertTrue(np.array_equal(periodicity, [False, True, True]))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ExceptionTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RadiiTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(DimensionalityTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(PeriodicFinderTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SearchGraphTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(DelaunayTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(AtomTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Class0DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Class1DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Material2DTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Class3DTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=3).run(alltests)
