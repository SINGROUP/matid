"""
Defines a set of regressions tests that should be run succesfully before
anything is pushed to the central repository.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import sys

import numpy as np
from ase import Atoms
from ase.visualize import view
import json

from systax import Classifier
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


def get_atoms(filename):
    with open(filename, "r") as fin:
        data = json.load(fin)
    pos = data["positions"]
    cell = data["normalizedCell"]
    num = data["labels"]

    atoms = Atoms(
        scaled_positions=pos,
        cell=1e10*np.array(cell),
        symbols=num,
        pbc=True
    )

    return atoms


class Class2DTests(unittest.TestCase):
    """Tests for the Class2D systems found in the NOMAD Archve for Exciting and
    FHIAims.
    """
    # def test_1(self):
        # system = get_atoms("./exciting2/Class2D/C4B2F4N2.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_2(self):
        # system = get_atoms("./exciting2/Class2D/C12H10N2.json")
        # # view(system)

        # classifier = Classifier(max_cell_size=30)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_3(self):
        # """This is an amorphous surface.
        # """
        # system = get_atoms("./fhiaims5/Class2D/Ba16Ge20O56.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_4(self):
        # """Looks like a surface, but the cell cannot be found. There is only
        # one layer.
        # """
        # system = get_atoms("./fhiaims6/Class2D/Ba16O40Si12.json")
        # # view(system)

        # classifier = Classifier(max_cell_size=12, pos_tol=0.9)
        # classification = classifier.classify(system)
        # print(classification)
        # self.assertIsInstance(classification, Class2D)

    # def test_5(self):
        # """Too sparse for 2D-material.
        # """
        # system = get_atoms("./fhiaims5/Class2D/F2.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_6(self):
        # """The position tolerance is too big to detect this properly. A too
        # simple cell is identified. When lowering the position tolerance the
        # structure is correctly classified.
        # """
        # system = get_atoms("./fhiaims6/Class2D/Ge12Mg12O36.json")

        # classifier = Classifier(pos_tol=0.3)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # additionals = classification.additional_indices
        # self.assertEqual(len(additionals), 0)

    # def test_7(self):
        # """The position tolerance is too big to detect this properly. A too
        # simple cell is identified. When lowering the position tolerance the
        # structure is correctly classified.
        # """
        # system = get_atoms("./fhiaims5/Class2D/Mg12O36Ti12.json")
        # # view(system)

        # classifier = Classifier(pos_tol=0.3)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

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
        # """Too sparse
        # """
        # system = get_atoms("./fhiaims5/Class2D/Ne2.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_8(self):
        # """Too sparse
        # """
        # system = get_atoms("./fhiaims5/Class2D/Ne2.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_9(self):
        # """Should be a pristine surface. Previously the max cell size was too
        # small.
        # """
        # system = get_atoms("./fhiaims5/Surface/Adsorbate/Ba16O48Si16.json")
        # # view(system)

        # classifier = Classifier(max_cell_size=12)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

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

    # def test_9(self):
        # """All the adsorbates were not correctly identified. Increasing the
        # similarity threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims5/Surface/Adsorbate/C2Ba16O44Zr12.json")
        # # view(system)

        # classifier = Classifier(max_cell_size=12)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns

        # # Print adsorbates
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 6)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)
        # self.assertTrue(set(adsorbates) == set((68, 69, 70, 71, 72, 73)))

    # def test_10(self):
        # """This system is basically a very thick surface with one layer only.
        # Thus it is not classified as surface (only one layer) nor Material2D
        # (too thick).
        # """
        # system = get_atoms("./fhiaims5/Class2D/Ca32O80Zr24.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)

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

    # def test_11(self):
        # """The adsorbate in this system was not detected for an unknown reason.
        # The bug was caused by to multiple overlapping edges in the periodicity
        # graph.
        # """
        # system = get_atoms("./fhiaims5/Surface/Pristine/CCa16O16.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 0)
        # self.assertEqual(len(adsorbates), 1)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_12(self):
        # """Adsorbate was added to the cell for unknown reason. This bug was
        # caused by too low threshold for filtering graphs with size different
        # from the graphs where the seed atom is.
        # """
        # system = get_atoms("./fhiaims5/Surface/Vacancy+Adsorbate/H4Mg12O14.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)

        # # Pristine
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

    # def test_13(self):
        # """In this system the unit cell has duplicate entries for two atoms.
        # Reason is still unknown, but does not affect the search because they
        # are so close.
        # """
        # system = get_atoms("./fhiaims5/Surface/Pristine/Ca24O88Zr32.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)

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
        # """Wrong vacancy detected, wrong adsorbate, wrong interstitial. Cause
        # by a bug in wrapping coordinates during the region tracking.
        # """
        # system = get_atoms("./fhiaims5/Surface/Vacancy+Interstitial+Adsorbate/Mg12O36Si12.json")
        # # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

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
        # """Wrong vacancy detected, wrong adsorbate, wrong interstitial. The
        # position tolerance needs to be modified and the chemical environment
        # tolerance relaxed for correct classification.
        # """
        # system = get_atoms("./fhiaims5/Surface/Vacancy+Interstitial+Adsorbate/H2Mg61NiO62.json")
        # view(system)

        # # classifier = Classifier()
        # classifier = Classifier(pos_tol=0.65, chem_similarity_threshold=0.3)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 1)
        # self.assertEqual(len(adsorbates), 2)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)
        # self.assertTrue(set(adsorbates) == set((124, 125)))

    # def test_16(self):
        # """Test the substitute detection on surfaces. All substitutes should be
        # validated based on the chemical environment instead of the tesselation.
        # """
        # system = get_atoms("./fhiaims5/Surface//Adsorbate/CMg39NiO40.json")
        # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Adsorbate + Substitution
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 1)
        # self.assertEqual(len(adsorbates), 1)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)
        # self.assertTrue(set(adsorbates) == set([80]))
        # self.assertTrue(substitutions[0].index == 65)

    # def test_17(self):
        # """The whole surface is not correctly detected. Increasing the
        # threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims5/Surface/Interstitial+Adsorbate/O80Sr32Zr24.json")
        # view(system)

        # classifier = Classifier(pos_tol=0.6)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

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

    # def test_17(self):
        # """The whole surface is not correctly detected. Increasing the
        # threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims5/Surface/Vacancy+Adsorbate/Mg20O68Si24.json")
        # view(system)

        # classifier = Classifier(pos_tol=0.6)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

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

    # def test_18(self):
        # """The whole surface is not correctly detected. Increasing the
        # threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims6/Surface/Vacancy+Interstitial+Substitution+Adsorbate/H2Mg61NiO62.json")
        # view(system)

        # classifier = Classifier(pos_tol=0.9)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # adsorbates = classification.adsorbates
        # interstitials = classification.interstitials
        # substitutions = classification.substitutions
        # vacancies = classification.vacancies
        # unknowns = classification.unknowns
        # self.assertEqual(len(vacancies), 0)
        # self.assertEqual(len(substitutions), 1)
        # self.assertEqual(len(adsorbates), 2)
        # self.assertEqual(len(unknowns), 0)
        # self.assertEqual(len(interstitials), 0)

    # def test_19(self):
        # """The whole surface is not correctly detected. Increasing the
        # threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims6/Surface/Interstitial+Adsorbate/O80Sr32Zr24.json")
        # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # additionals = classification.additional_indices()
        # self.assertEqual(len(additionals), 0)

    # def test_20(self):
        # """The whole surface is not correctly detected. Increasing the
        # threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims6/Surface/Interstitial+Adsorbate/CMg24O74Si24.json")
        # view(system)

        # classifier = Classifier(pos_tol=0.65)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # 11 additional atoms
        # additional_indices = classification.additional_indices
        # # print(additional_indices)
        # self.assertEqual(len(additional_indices), 11)
        # self.assertTrue(set(additional_indices) == set([120, 121, 122,  57, 87, 27, 117, 3, 33, 63, 93]))

    # def test_21(self):
        # """The whole surface is not correctly detected. Increasing the
        # threshold fixes the problem.
        # """
        # system = get_atoms("./fhiaims6/Surface/Interstitial+Adsorbate/Ca32O80Ti24.json")
        # view(system)

        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # additionals = classification.additional_indices
        # self.assertEqual(len(additionals), 0)

    # def test_22(self):
        # """This is a surface with only one repetition of the unit cell. Should
        # be left as Class2D. Fixed by increasing the requirement for the number
        # of edges for a graph corrresponding to a span to 1.0.
        # """
        # system = get_atoms("./fhiaims6/Surface/Interstitial+Adsorbate/C8Mo16.json")
        # view(system)

        # classifier = Classifier(max_cell_size=14, pos_tol=0.2)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Class2D)

    # def test_23(self):
        # """The default selection of the seed point near the center of the cell
        # does not work for this surface, because the middle atom is not repeated
        # enough times. By manually selecting the seed point the correct surface
        # is found.
        # """
        # system = get_atoms("./fhiaims8/Surface/Ca20O48Zr16+Ca12O32Zr8.json")
        # view(system)

        # classifier = Classifier(seed_position=71)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # additionals = classification.additional_indices
        # self.assertEqual(len(additionals), 0)

    # def test_24(self):
        # system = get_atoms("./fhiaims8/Surface/Ca20O48Zr16+Ca12O32Zr8.json")
        # view(system)

        # classifier = Classifier(seed_position=71)
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # additionals = classification.additional_indices
        # self.assertEqual(len(additionals), 0)

    # def test_25(self):
        # system = get_atoms("./fhiaims8/Surface/Mg61O55+H2NiO7.json")
        # view(system)

        # # classifier = Classifier(seed_position=12)
        # classifier = Classifier()
        # classification = classifier.classify(system)
        # self.assertIsInstance(classification, Surface)

        # # Pristine
        # additionals = classification.additional_indices
        # self.assertEqual(len(additionals), 0)

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(Class2DTests))

    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
