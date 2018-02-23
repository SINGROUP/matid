# import unittest

# import numpy as np

# from systax.tools.surfacegenerator import SurfaceGenerator
# from systax import Material3DAnalyzer
# from systax import Classifier
# from systax.classification import Surface

# from ase.lattice.cubic import FaceCenteredCubic


# class SurfaceTests(unittest.TestCase):

    # def test_surfaces(self):
        # atoms = FaceCenteredCubic(
            # directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            # size=(1, 1, 1),
            # symbol='Cu',
            # pbc=(1, 1, 1)
        # )

        # structures = [atoms]
        # directions = [(1, 0, 0)]
        # layers = 3
        # vacuum = 5

        # for structure in structures:
            # for direction in directions:

                # # Create conventional representation
                # analyzer = Material3DAnalyzer(structure)
                # conv_sys = analyzer.get_conventional_system()
                # wyckoff_letters = analyzer.get_wyckoff_letters_conventional()

                # # Generate surface
                # gen = SurfaceGenerator()
                # surface = gen.generate(conv_sys, direction, layers, vacuum)

                # # Classify
                # classifier = Classifier()
                # classification = classifier.classify(surface)

                # self.assertIsInstance(classification, Surface)

                # # Detect conventional representation
                # surface = classification.surfaces[0]
                # analyzer = surface.bulk_analyzer
                # conv_sys_det = analyzer.get_conventional_system()
                # wyckoff_letters_det = analyzer.get_wyckoff_letters_conventional()

                # # Compare original and detected conventional system
                # self.assertTrue(np.allclose(conv_sys.get_cell(), conv_sys_det.get_cell()))
                # self.assertTrue(np.allclose(conv_sys.get_positions(), conv_sys_det.get_positions()))
                # self.assertTrue(np.allclose(conv_sys.get_atomic_numbers(), conv_sys_det.get_atomic_numbers()))
                # self.assertTrue(np.array_equal(wyckoff_letters, wyckoff_letters_det))

                # # Detect Miller index
                # surf_analyzer = surface.analyzer
                # miller_detected = surf_analyzer.get_miller_indices()
                # print(miller_detected)

                # # Compare original and detected miller indices

# if __name__ == '__main__':
    # suites = []
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SurfaceTests))
    # alltests = unittest.TestSuite(suites)
    # result = unittest.TextTestRunner(verbosity=0).run(alltests)
