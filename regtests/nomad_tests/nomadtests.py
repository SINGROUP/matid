"""
Defines a set of regressions tests that should be run succesfully before
anything is pushed to the central repository.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import sys

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
import ase.io
import json

from systax import Classifier
from systax import PeriodicFinder
from systax.classification import \
    Class0D, \
    Class1D, \
    Class2D, \
    Class3D, \
    Atom, \
    Molecule, \
    Crystal, \
    Material1D, \
    Material2D, \
    Unknown, \
    Surface
from systax import Class3DAnalyzer
from systax.data.constants import WYCKOFF_LETTER_POSITIONS
import systax.geometry


class Class2DTests(unittest.TestCase):
    """Tests for the Class2D systems found in the NOMAD Archve for Exciting and
    FHIAims.
    """
    # def test_1(self):
        # with open("./PKPif9Fqbl30oVX-710UwCHGMd83y.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Material2D)

    # def test_4(self):
        # with open("./P8Wnwz4dfyea6UAD0WEBadXv83wyf.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

    # def test_5(self):
        # with open("./PEzXqLISX8Pam-HlJMxeLc86lcKgf.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Only adsorbates
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 14)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_6(self):
        # """Surface with a big (>8 angstrom) basis vector in the unit cell.
        # """
        # with open("./Pwf1I3LmgToiTGVzGWuPGMsk8qhG2.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # print(len(system))
        # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_7(self):
        # with open("./Pm1yla_i8HghYOCx4zYghsccncpYb.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_8(self):
        # with open("./P6ify4HgqDkkettDovKwl7_A9emhy.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # print(classification)

        # # Adsorbates
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 6)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_9(self):
        # """
        # """
        # with open("./PgFB5vtxkTyEJ3oUZ0ylWn0A-z8ke.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # print(classification)

        # # Adsorbates
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 6)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_10(self):
        # """
        # """
        # with open("./PnG-oRNfRLg1L4Veolbdbqr16SAZT.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # print(classification)

        # # Adsorbates
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 30)
        # self.assertTrue(np.array_equal(adsorbates, range(30)))
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_11(self):
        # """
        # """
        # with open("./Pje1-YdvEDusyBIStylnee37oeiPW.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # print(classification)

        # # Adsorbates
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 8)
        # self.assertTrue(np.array_equal(adsorbates, range(8)))
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_12(self):
        # """
        # """
        # with open("./Pt08E8pJfZPLEzeiCLDIf7lyNv9lX.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_13(self):
        # """
        # """
        # with open("./PSfyWNLc5i4SvDGeYHA2vAtzFEpgn.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()  # Fixed by pos tol 0.6
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_14(self):
        # """
        # """
        # with open("./PhXWhMLAoqLBzANjPHKYReu9WrE2k.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # view(system)

        # classifier = Classifier()  # Works with pos_tol = 0.6
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_15(self):
        # """
        # """
        # with open("./P1re6oFZ3KTCcl9CID_P1lu00yN5I.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_16(self):
        # """
        # """
        # with open("./PEd7LZB9mmpNk1hKQINX0tvHq3wKg.json", "r") as fin:
            # data = json.load(fin)

        # section_system = data["sections"]["section_run-0"]["sections"]["section_system-0"]

        # system = Atoms(
            # positions=1e10*np.array(section_system["atom_positions"]),
            # cell=1e10*np.array(section_system["simulation_cell"]),
            # symbols=section_system["atom_labels"],
            # pbc=True,
        # )
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)
        # # # print(classification)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 0)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(NomadTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
