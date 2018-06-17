from matid import Classifier
from ase import Atoms

# Define the structure as an ASE Atoms object
system = Atoms(
    positions=[
        [5., 5.,       5.2981545],
        [5., 5.763239, 4.7018455],
        [5., 4.236761, 4.7018455],
    ],
    symbols=["O", "H", "H"],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

# Define the classifier
classifier = Classifier()

# Perform classification
classification = classifier.classify(system)

# Investigate result
print(classification)
