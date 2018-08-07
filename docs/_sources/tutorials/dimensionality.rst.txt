Dimensionality Detection
========================

This tutorial showcases how the package can be used to find out the
dimensionality of any arbitrary atomistic geometry.

To determine the dimensionality of a system, we use a modified version of the
topological scaling algorithm (TSA) [1]_. The algorithm is based on analyzing
the size scaling of atomic clusters when going from the original system to a
bigger supercell of the same system. With TSA, the dimensionality :math:`D` is
given by

.. math::

   D=\begin{cases}
      n_\text{pbc}-\log_n (N_{n}) \text{, when}~n_\text{pbc} \neq 0  \\
      0\text{, when}~n_\text{pbc} = 0
   \end{cases}

where :math:`N_n` is the number of clusters in a supercell that is repeated
:math:`n` times in each periodic direction and :math:`n_\mathrm{pbc}` is the
number of periodic dimensions. For the clustering we use the Density-Based
Spatial Clustering of Applications with Noise [2]_ data clustering algorithm.
The advantage of this algorithm is that it does not require an initial guess
for the number of clusters and it can find arbitrarily shaped clusters. The
clustering requires that we define a metric for the distance between atoms. We
use the following metric:

.. math::

   d_{ij} = \lvert \vec{r}_i - \vec{r}_j \rvert^{\text{MIC}} - r_i^{\text{cov}} - r_j^{\text{cov}}

where :math:`\vec{r}_i` and :math:`\vec{r}_i` are the cartesian positions of
atom :math:`i` and :math:`j`, respectively, and :math:`r_i^{\text{cov}}` and
:math:`r_j^{\text{cov}}` their covalent radii [3]_ . It is important to
notice that in this metric the distances always follow the minimum image
convention (MIC), i.e.  the distance is calculated between two closest periodic
neighbours. By using the distance to the closest periodic neighbour we obtain
the correct clusters regardless of what shape of cell is used in the original
simulation.

The clustering uses two parameters: the minimum cluster size
:math:`n_\mathrm{min}` and the neighbourhood radius :math:`\epsilon`. We set
:math:`n_\mathrm{min}` to 1 to allow clusters consisting of even single atoms
and :math:`\epsilon` defaults to 3.5 Å. At present, a system, in which there is
more than one cluster in the original non-repeated system (:math:`N_1 \gt 1`),
is classified as unknown. Such a case corresponds to systems with multiple
components that are spatially separated, such as a molecule far above a
surface, low density gases, widely spaced clusters in vacuum, etc.

The following code illustrates how dimensionality detection can be performed
with MatID.

.. code-block:: python

   from matid.geometry import get_dimensionality

   from ase.build import molecule
   from ase.build import nanotube
   from ase.build import mx2
   from ase.build import bulk

   # Here we create one example of each dimensionality class
   zero_d = molecule("H2O", vacuum=5)
   one_d = nanotube(6, 0, length=4, vacuum=5)
   two_d = mx2(vacuum=5)
   three_d = bulk("NaCl", "rocksalt", a=5.64)

   # In order to make the dimensionality detection interesting, we add periodic
   # boundary conditions. This is more realistic as not that many electronic
   # structure codes support anything else than full periodic boundary conditions,
   # and thus the dimensionality information is typically not available.
   zero_d.set_pbc(True)
   one_d.set_pbc(True)
   two_d.set_pbc(True)
   three_d.set_pbc(True)

   # Here we perform the dimensionality detection with clustering threshold eta
   epsilon = 3.5
   dim0 = get_dimensionality(zero_d, epsilon)
   dim1 = get_dimensionality(one_d, epsilon)
   dim2 = get_dimensionality(two_d, epsilon)
   dim3 = get_dimensionality(three_d, epsilon)

   # Printing out the results
   print(dim0)
   print(dim1)
   print(dim2)
   print(dim3)

This example if also available in "examples/dimensionality.py".

.. [1] Ashton, M., Paul, J., Sinnott, S. B. & Hennig, R. G. Topology-scaling identification of layered solids and stable exfoliated 2d materials. Phys. Rev. Lett. 118, 106101 (2017)

.. [2] Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. A density-based algorithm for discovering clusters in large spatial databases with noise. KDD’96 Proceedings of the Second International Conference on Knowledge Discovery and Data Mining 226–231 (1996).

.. [3] Cordero, B. et al. Covalent radii revisited. Dalton Trans. 2832–2838 (2008)
