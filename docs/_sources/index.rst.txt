MatID
=====

.. image:: https://travis-ci.org/SINGROUP/matid.svg?branch=master
    :target: https://travis-ci.org/SINGROUP/matid

.. image:: https://coveralls.io/repos/github/SINGROUP/matid/badge.svg?branch=master
    :target: https://coveralls.io/github/SINGROUP/matid?branch=master

MatID is a python package for identifying and analyzing atomistic systems based
on their structure. MatID is designed to help researchers in the automated
analysis and labeling of atomistic datasets.

Capabilities at a Glance
========================

With MatID you can:

  - :doc:`Automatically analyze structural features in a dataset <tutorials/overview>`
  - :doc:`Automatically classify atomic geometries into different structural classes <tutorials/classification>`
  - Automatically identify outlier atoms such as adsorbates in surfaces geometries (tutorial in development)
  - :doc:`Determine the dimensionality of an atomistic object <tutorials/dimensionality>`
  - Analyze symmetry properties of 3D structures (tutorial in development)

Check the tutorials to see more information.

Go Deeper
=========

Documentation for the source code :doc:`can be found here <doc/modules>`. The
full source code with examples and regression tests can be explored at `github
<https://github.com/SINGROUP/matid>`_.

.. toctree::
    :hidden:

    install
    tutorials/tutorials
    Documentation <doc/modules>
    about

Cite
====
If you found MatID useful in your research, please cite:
`Himanen, L. and Rinke, P. and Foster, A. S., Materials structure genealogy and high-throughput topological classification of surfaces and 2D materials, npj Comput. Mater. 4, 52, (2018) <http://www.nature.com/articles/s41524-018-0107-6>`_

BibTex entry:

.. code-block:: none

   @article{matid,
      author = {Himanen, Lauri and Rinke, Patrick and Foster, Adam Stuart},
      title = {{Materials structure genealogy and high-throughput topological classification of surfaces and 2D materials}},
      journal = {npj Computational Materials},
      volume = {4},
      number = {52},
      year = {2018}
      publisher = {Springer US},
      doi = {10.1038/s41524-018-0107-6},
      url = {http://www.nature.com/articles/s41524-018-0107-6},
   }
