import numpy as np
from ase.visualize import view
from ase.build import bcc100, molecule
from matid import Classifier, SymmetryAnalyzer
import ase.io

# Generating a surface adsorption geometry with ASE.
adsorbent = bcc100('Fe', size=(3, 3, 4), vacuum=8)
ase.io.write("structure.xyz", adsorbent)

# adsorbate = molecule("H2O")
# adsorbate.rotate(180, [1, 0, 0])
# adsorbate.translate([4.3, 4.3, 13.5])
# system = adsorbent + adsorbate
# system.set_pbc([True, True, True])

# # Add noise and defects to the structure
# positions = system.get_positions()
# positions += 0.25*np.random.rand(*positions.shape)
# system.set_positions(positions)
# del system[31]

# # Visualize the final system
# view(system)

# # Run the classification
# classifier = Classifier(pos_tol=1.0, max_cell_size=6)
# classification = classifier.classify(system)

# # Print classification
# print("Structure classified as: {}".format(classification))

# # Print found outliers
# outliers = classification.outliers
# print("Outlier atoms indices: {}".format(outliers))

# # Visualize the cell that was found by matid
# prototype_cell = classification.prototype_cell
# view(prototype_cell)

# # Visualize the corresponding conventional cell
# analyzer = SymmetryAnalyzer(prototype_cell, symmetry_tol=0.5)
# conv_sys = analyzer.get_conventional_system()
# view(conv_sys)

# # Visualize the corresponding primitive cell
# prim_sys = analyzer.get_primitive_system()
# view(prim_sys)

# # Print space group number
# spg_number = analyzer.get_space_group_number()
# print("Space group number: {}".format(spg_number))
