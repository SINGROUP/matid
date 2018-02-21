```text
               _
              | |
 ___ _   _ ___| |_ __ ___  __
/ __| | | / __| __/ _` \ \/ /
\__ \ |_| \__ \ || (_| |>  <
|___/\__, |___/\__\__,_/_/\_\
      __/ |
     |___/
```
Systax is a python package for the analysis and classification of atomistic
systems.

Systax can be used e.g. to perform the following tasks:

    - Given an atomistic system, automatically classify it to one of the
      predetermined system types, such as surface, molecule, atom, crystal, ..
    - Given a crystal structure, get the idealized primitive or conventional
      system.
    - Given a surface system, get the idealized bulk cell corresponding to the
      surface.

Example usage:
```python
import ase.io
from systax import classifier, Surface

# Read a geometry from file with ASE and ensure that the cell and periodicity
are given.
system = ase.io.read("geometry.xyz")
system.set_cell([20, 20, 20])
system.set_pbc(True)

# Run the classification and if surface matched, get the detected unit cell and
indices of outlier atoms
classifier = Classifier()
classification = classifier.classify(system)

if type(classification) = Surface:
    cell = classification.cell
    outliers = classification.outliers
```
