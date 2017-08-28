from ase.build import bcc100, add_adsorbate
from ase.visualize import view
from ase.build import molecule

from systax import Classifier

# Create an Fe 100 surface as an ASE Atoms object
system = bcc100('Fe', size=(4, 4, 3), vacuum=8)
system.rattle(0.05)
view(system)

# Analyze the system
classifier = Classifier(pos_tol=0.5)
classifier.classify(system)

# Get the surface in the system
surfaces = classifier.surfaces
surface = surfaces[0]
# print(surface.get_normalized_cell())
# print(surface.original_indices)
