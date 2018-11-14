from matid import Classifier
import ase.io

# Read an extended XYZ file containin an atomic geometry. Extended XYZ files
# will also include the unit cell and periodic boundary conditions.
system = ase.io.read("structure.xyz")

# Define the classifier
classifier = Classifier()

# Perform classification
classification = classifier.classify(system)

# Investigate result
print(classification)
