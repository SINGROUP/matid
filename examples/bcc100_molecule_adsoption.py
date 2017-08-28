from ase.build import bcc100, add_adsorbate
from ase.visualize import view
from ase.build import molecule

from systax import Classifier

# Create an Fe 100 surface as an ASE Atoms object
system = bcc100('Fe', size=(4, 4, 3), vacuum=8)
molecule = molecule("C6H6")
add_adsorbate(system, molecule, height=2, offset=(2, 2.5))
# view(system)

# Analyze the system
classifier = Classifier()
classifier.classify(system)

# Get the surface in the system
surfaces = classifier.surfaces
surface = surfaces[0]
# print(surface.get_normalized_cell())
# print(surface.original_indices)
