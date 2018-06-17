.. _classification:

Classification
==============

MatID can be used to automatically classify structures into different
structural classes. The currently supported classes are:

  - Class3D: Generic class for all three-dimensional structures, e.g. bulk
    crystals, liquids, solid-liquid interfaces.
  - Class2D: Generic class for all two-dimensional structures, e.g. surfaces, 2D materials.

    - Surface: Specific class for surface structures.
    - Material2D: Specific class for 2D materials.

  - Class1D: Generic class for all one-dimensional structures, e.g.
    polymers, nanotubes.
  - Class0D: Generic class for all zero-dimensional structures, e.g.
    atoms, molecules, clusters.

The classification is based on the structural properties of the system, i.e.
atomic positions, unit cell, periodic boundary conditions and atomic numbers.
The classification, like the whole MatID package supports the ASE library for
handling atomic strucures. With ASE you can read structures from multiple
files, or define the structure yourself with the Atoms class or the different
structure creation tools. Once the structure has been specified as an ASE Atoms
object, you can input it into classification.

The following code shows an example of reading a structure from an extended XYZ
file and classifying it.

.. literalinclude:: ../../examples/classification1.py
   :language: python

An alternative way is to define the Atoms object yourself

.. literalinclude:: ../../examples/classification2.py
   :language: python
