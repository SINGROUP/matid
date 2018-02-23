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

# Example: Automatic classification of atomic systems

```python
import ase.io
from systax import classifier, Surface

# Read a geometry from file with ASE and ensure that the cell and periodicity
# are given.
system = ase.io.read("geometry.xyz")
system.set_cell([20, 20, 20])
system.set_pbc(True)

# Run the classification and if surface matched, get the detected unit cell and
# indices of outlier atoms
classifier = Classifier()
classification = classifier.classify(system)

if type(classification) = Surface:
    cell = classification.cell
    outliers = classification.outliers
```

