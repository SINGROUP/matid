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

# Here we perform the dimensionality detection with clustering threshold epsilon
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
