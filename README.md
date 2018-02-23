# Systax
Systax is a python package for the structural analysis and classification of
atomistic systems.

# Installation
The package is compatible with Python 2.7, 3.4-3.6, and the dependencies are
given in the file 'requirements.txt. These dependencies will be automatically
installed during the setup of the package.

The latest stable release is available through pip, and can be installed with
the command:
```sh
pip install systax
```

To install the most recent development version, you can clone this repository
and perform the install with pip directly from the source code:

```sh
git clone https://gitlab.com/laurih/systax.git
cd systax
pip install .
```

# Example: Surface detection and analysis

```python
import ase.io
from ase.visualize import view
from systax import Classifier
from systax.classifications import Surface

# Read a geometry from file with ASE. Ensure that the cell and periodicity are
# correctly given for the structure.
system = ase.io.read("geometry.xyz")
view(system)

# Run the classification
classifier = Classifier(pos_tol=0.5, max_cell_size=6)
classification = classifier.classify(system)

if type(classification) == Surface:

    # View the conventional cell corresponding to this surface
    conventional_cell = classification.conventional_cell
    view(conventional_cell)

    # View the outlier atoms
    outliers = classification.outliers
    view(system[outliers])

    # Inspect more symmetry details about the cell
    analyzer = classification.cell_analyzer
    space_group = analyzer.get_space_group_number()
    print(space_group)

    primitive_cell = analyzer.get_primitive_system()
    view(primitive_cell)
```

