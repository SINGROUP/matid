Analyzing a dataset
===================

This tutorial introduces the basic functionality of the package when applied to
a real-world analysis of a dataset containing atomic structures.

Lets start by loading a series of geometries as `ASE.Atoms
<https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_ objects. In this case we have
used a list of extended xyz-files to represent our dataset. But in general as
long as the atomic positions, unit cell, periodic boundary conditions and
chemical symbols for the atoms are available, any dataset can be turned into an
ASE.Atoms object and analyzed by MatID.

So let's start by loading a series of structures into a list:

.. literalinclude:: ../../../examples/summary.py
   :lines: 1-22

With a list of geometries available, we can start analyzing them with MatID.
Typically the first task is to get a generic classification for the structure.
This is done with the Classifier-class:

.. literalinclude:: ../../../examples/summary.py
   :lines: 24-33

Once we have determined the structural class of each geometry, we can further
query for additional information that depends on the detected classification,
and create a summary of the results:

.. literalinclude:: ../../../examples/summary.py
   :lines: 35-

You can find the full example in "examples/summary.py". Here are the results:

.. raw:: html

    <div class="table">
        <div class="tableheader">
            <p id="namecol">Filename</p><p>Results</p><p>Image</p>
        </div>
        <div class="tablerow">
            <p>C32Mo32+CO2.xyz</p>
            <p>
            system_type: Surface<br/>
            outlier_formula: CO2<br/>
            outlier_indices: [64, 65, 66]<br/>
            space_group_number: 225<br/>
            crystal_system: cubic<br/>
            bravais_lattice: cF
            </p>
            <img id="resultimg" src="../_static/img/C32Mo32+CO2.jpg">
        </div>
        <div class="tablerow">
            <p>C49+N.xyz</p>
            <p>
            system_type: Material2D<br/>
            outlier_formula: N<br/>
            outlier_indices: [49]<br/>
            </p>
            <img id="resultimg" src="../_static/img/C49+N.jpg">
        </div>
        <div class="tablerow">
            <p>H2O.xyz</p>
            <p>
            system_type: Class0D<br/>
            </p>
            <img id="resultimg" src="../_static/img/H2O.jpg">
        </div>
        <div class="tablerow">
            <p>Si8.xyz</p>
            <p>
            system_type: Class3D<br/>
            space_group_number: 227<br/>
            crystal_system: cubic<br/>
            bravais_lattice: cF
            </p>
            <img id="resultimg" src="../_static/img/Si8.jpg">
        </div>
        <div class="tablerow">
            <p>Mg61O62+CH4Ni.xyz</p>
            <p>
            system_type: Surface<br/>
            outlier_formula: CH4Ni<br/>
            outlier_indices: [72, 124, 125,126, 127, 128]<br/>
            space_group_number: 225<br/>
            crystal_system: cubic<br/>
            bravais_lattice: cF
            </p>
            <img id="resultimg" src="../_static/img/Mg61O62+CH4Ni.jpg">
        </div>
        <div class="tablerow">
            <p>C26H24N4O2.xyz</p>
            <p>
            system_type: Class2D
            </p>
            <img id="resultimg" src="../_static/img/C26H24N4O2.jpg">
        </div>
        <div class="tablerow">
            <p>Ru.xyz</p>
            <p>
            system_type: Atom
            </p>
            <img id="resultimg" src="../_static/img/Ru.jpg">
        </div>
    </div
