.. _classification:

Classification
==============

MatID can be used to automatically classify structures into different
structural classes. The currently supported classes are:

  - :class:`.Class3D`: Generic class for all three-dimensional structures, e.g. bulk
    crystals, liquids, solid-liquid interfaces.
  - :class:`.Class2D`: Generic class for all two-dimensional structures, e.g. surfaces, 2D materials.

    - :class:`.Surface`: Specific class for surface structures.
    - :class:`.Material2D`: Specific class for 2D materials.

  - :class:`.Class1D`: Generic class for all one-dimensional structures, e.g.
    polymers, nanotubes.
  - :class:`.Class0D`: Generic class for all zero-dimensional structures, e.g.
    atoms, molecules, clusters.

    - :class:`.Atom`: Specific class for single atoms.

The classification system is hierarchical: e.g. a Surface is a subclass of a
Class2D. If a two-dimensional structure cannot be assigned with certainty into a
more specific subclass, the most specific applicable parent class is used.

The classification is based on the structural properties of the system, i.e.
atomic positions, unit cell, periodic boundary conditions and atomic numbers.
The classification, like the whole MatID package supports the `ASE library
<https://wiki.fysik.dtu.dk/ase/>`_ for handling atomic structures. With ASE you
can read structures from multiple files, or define the structure yourself with
the Atoms class or the different structure creation tools. Once the structure
has been specified as an `ASE.Atoms
<https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_ object, you can input it into
classification.

The following code shows an example of reading a structure from an extended XYZ
file and classifying it.

.. literalinclude:: ../../../examples/classification1.py
   :language: python

An alternative way is to define the Atoms object yourself

.. literalinclude:: ../../../examples/classification2.py
   :language: python

The :class:`.Classifier` class provides multiple parameters that can be used to tune the
classification to fit different scenarios. The default settings are quite
conservative: they are good for a wide range of systems, but can result in slow
classification for bigger systems. The speed can be improved by providing
custom settings if more is known about the dataset.
