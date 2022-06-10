# MatID

![Build status](https://github.com/SINGROUP/matid/actions/workflows/build.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/SINGROUP/matid/badge.svg?branch=master)](https://coveralls.io/github/SINGROUP/matid?branch=master)

MatID is a python package for identifying and analyzing atomistic systems based
on their structure.

## Homepage
For more details and tutorials, visit the homepage at:
[https://singroup.github.io/matid/](https://singroup.github.io/matid/)

## Installation
The newest versions of the package are compatible with Python >= 3.7 (tested on
3.7, 3.8, 3.9 and 3.10). MatID versions <= 0.5.4 also support Python 2.7. The
exact list of dependencies are given in setup.py and all of them will be
automatically installed during setup.

The latest stable release is available through pip: (use the -\\-user flag if
root access is not available)

```sh
    pip install matid
```

To install the latest development version, clone the source code from github
and install with pip from local file:

```sh
    git clone https://github.com/SINGROUP/matid.git
    cd matid
    pip install .
```

## Example: Surface detection and analysis

```python
import numpy as np
from ase.visualize import view
from ase.build import bcc100, molecule
from matid import Classifier, SymmetryAnalyzer

# Generating a surface adsorption geometry with ASE.
adsorbent = bcc100('Fe', size=(3, 3, 4), vacuum=8)
adsorbate = molecule("H2O")
adsorbate.rotate(180, [1, 0, 0])
adsorbate.translate([4.3, 4.3, 13.5])
system = adsorbent + adsorbate
system.set_pbc([True, True, True])

# Add noise and defects to the structure
positions = system.get_positions()
positions += 0.25*np.random.rand(*positions.shape)
system.set_positions(positions)
del system[31]

# Visualize the final system
view(system)

# Run the classification
classifier = Classifier(pos_tol=1.0, max_cell_size=6)
classification = classifier.classify(system)

# Print classification
print("Structure classified as: {}".format(classification))

# Print found outliers
outliers = classification.outliers
print("Outlier atoms indices: {}".format(outliers))

# Visualize the cell that was found by matid
prototype_cell = classification.prototype_cell
view(prototype_cell)

# Visualize the corresponding conventional cell
analyzer = SymmetryAnalyzer(prototype_cell, symmetry_tol=0.5)
conv_sys = analyzer.get_conventional_system()
view(conv_sys)

# Visualize the corresponding primitive cell
prim_sys = analyzer.get_primitive_system()
view(prim_sys)

# Print space group number
spg_number = analyzer.get_space_group_number()
print("Space group number: {}".format(spg_number))
```
