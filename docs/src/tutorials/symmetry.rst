Symmetry analysis
===================

MatID has extensive symmetry analysis tools for 3D periodic systems. The
symmetry detection is powered by `spglib <https://atztogo.github.io/spglib/>`_,
but has been extended with additional analysis and caching functionality for
increased performance on closely related queries.

The basis of symmetry analysis is the :class:`.SymmetryAnalyzer`-class. It
takes as input an atomic geometry, and symmetry tolerance settings.

.. literalinclude:: ../../../examples/symmetry.py
   :lines: 1-11

The conventional and primitive systems corresponding to the given structure can
be directly queried as `ASE.Atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_-objects

.. literalinclude:: ../../../examples/symmetry.py
   :lines: 13-19

Further symmetry information can be queried as follows:

.. literalinclude:: ../../../examples/symmetry.py
   :lines: 24-49

Which will output the following:

.. code-block:: none

   Space group number: 225
   Space group international short symbol: Fm-3m
   Is chiral: False
   Hall number: 523
   Hall symbol: -F 4 2 3
   Crystal system: cubic
   Bravais lattice: cF
   Point group: m-3m
   Wyckoff letters original: ['a' 'b' 'a' 'b' 'a' 'b' 'a' 'b' 'a' 'b' 'a' 'b' 'a' 'b' 'a' 'b']
   Wyckoff letters primitive: ['a' 'b']
   Wyckoff letters conventional: ['a' 'b' 'a' 'b' 'a' 'b' 'a' 'b']

MatID also utilises offline information from the `Bilbao crystallographic
server <http://www.cryst.ehu.es/>`_ to analyze the detailed Wyckoff set
information for structures. With this information the details of the Wyckoff
sets contained in the structure can be analyzed. Here we demonstrate this
functionality on a more complex silicon clathrate structure.

.. literalinclude:: ../../../examples/symmetry.py
   :lines: 53-

Which will output the following information:

.. code-block:: none

   Set 0
      Letter: c
      Element: Si
      Indices: [40, 41, 42, 43, 44, 45]
      Multiplicity: 6
      Repr.: ['1/4', '0', '1/2']
      x: None
      y: None
      z: None
   Set 1
      Letter: i
      Element: Si
      Indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      Multiplicity: 16
      Repr.: ['x', 'x', 'x']
      x: 0.18369999999999995
      y: None
      z: None
   Set 2
      Letter: k
      Element: Si
      Indices: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
      Multiplicity: 24
      Repr.: ['0', 'y', 'z']
      x: None
      y: 0.30769999999999975
      z: 0.11719999999999986

You can find the full example in "examples/symmetry.py".
