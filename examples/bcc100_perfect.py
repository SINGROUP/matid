from __future__ import absolute_import, division, print_function, unicode_literals
from ase.build import bcc100
from ase.visualize import view
from systax import Classifier

# Create an Fe 100 surface as an ASE Atoms object
system = bcc100('Fe', size=(3, 3, 3), vacuum=8)
# view(system)

# Analyze the system
classifier = Classifier()
classifier.classify(system)

# Get the surface in the system
surfaces = classifier.surfaces
surface = surfaces[0]
bulk_cell = surface.get_normalized_cell()
print(bulk_cell.relative_pos)
print(bulk_cell.lattice.matrix)
print(bulk_cell.numbers)
print(bulk_cell.wyckoff_letters)
