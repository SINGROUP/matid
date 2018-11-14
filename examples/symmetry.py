from matid import SymmetryAnalyzer
from ase.build import bulk
from ase.visualize import view
from ase import Atoms

# Prepare a geometry to be analyzed
nacl = bulk("NaCl", "rocksalt", a=5.64).repeat([2, 2, 2])
view(nacl)

# Setup the symmetry analyzer
symm = SymmetryAnalyzer(nacl, symmetry_tol=0.1)

# Get the conventional system as an ase.Atoms-object.
conv = symm.get_conventional_system()
view(conv)

# Get the primitive system as an ase.Atoms-object
prim = symm.get_primitive_system()
view(prim)

# Get symmetry related information. Some properties depenc on which system is
# meant: the original, primitive or conventional system. To avoid ambiquity,
# the method names include for which system these properties are reported.
space_group_number = symm.get_space_group_number()
space_group_symbol = symm.get_space_group_international_short()
is_chiral = symm.get_is_chiral()
hall_number = symm.get_hall_number()
hall_symbol = symm.get_hall_symbol()
crystal_system = symm.get_crystal_system()
bravais_lattice = symm.get_bravais_lattice()
point_group = symm.get_point_group()
wyckoff_letters_orig = symm.get_wyckoff_letters_original()
wyckoff_letters_prim = symm.get_wyckoff_letters_primitive()
wyckoff_letters_conv = symm.get_wyckoff_letters_conventional()
equivalent_atoms_orig = symm.get_equivalent_atoms_original()
equivalent_atoms_prim = symm.get_equivalent_atoms_primitive()
equivalent_atoms_conv = symm.get_equivalent_atoms_conventional()

print("Space group number: {}".format(space_group_number))
print("Space group international short symbol: {}".format(space_group_symbol))
print("Is chiral: {}".format(is_chiral))
print("Hall number: {}".format(hall_number))
print("Hall symbol: {}".format(hall_symbol))
print("Crystal system: {}".format(crystal_system))
print("Bravais lattice: {}".format(bravais_lattice))
print("Point group: {}".format(point_group))
print("Wyckoff letters original: {}".format(wyckoff_letters_orig))
print("Wyckoff letters primitive: {}".format(wyckoff_letters_prim))
print("Wyckoff letters conventional: {}".format(wyckoff_letters_conv))

# Print out details of the Wyckoff sets contained in a more complex silicon
# clathrate structure.
scaled_positions = [
    [0.18370000000000, 0.18370000000000, 0.18370000000000],
    [0.18370000000000, 0.18370000000000, 0.81630000000000],
    [0.18370000000000, 0.81630000000000, 0.18370000000000],
    [0.18370000000000, 0.81630000000000, 0.81630000000000],
    [0.31630000000000, 0.31630000000000, 0.31630000000000],
    [0.31630000000000, 0.31630000000000, 0.68370000000000],
    [0.31630000000000, 0.68370000000000, 0.31630000000000],
    [0.31630000000000, 0.68370000000000, 0.68370000000000],
    [0.68370000000000, 0.31630000000000, 0.31630000000000],
    [0.68370000000000, 0.31630000000000, 0.68370000000000],
    [0.68370000000000, 0.68370000000000, 0.31630000000000],
    [0.68370000000000, 0.68370000000000, 0.68370000000000],
    [0.81630000000000, 0.18370000000000, 0.18370000000000],
    [0.81630000000000, 0.18370000000000, 0.81630000000000],
    [0.81630000000000, 0.81630000000000, 0.18370000000000],
    [0.81630000000000, 0.81630000000000, 0.81630000000000],
    [0.00000000000000, 0.11720000000000, 0.30770000000000],
    [0.00000000000000, 0.11720000000000, 0.69230000000000],
    [0.00000000000000, 0.88280000000000, 0.30770000000000],
    [0.00000000000000, 0.88280000000000, 0.69230000000000],
    [0.11720000000000, 0.30770000000000, 0.00000000000000],
    [0.11720000000000, 0.69230000000000, 0.00000000000000],
    [0.19230000000000, 0.38280000000000, 0.50000000000000],
    [0.19230000000000, 0.61720000000000, 0.50000000000000],
    [0.30770000000000, 0.00000000000000, 0.11720000000000],
    [0.30770000000000, 0.00000000000000, 0.88280000000000],
    [0.38280000000000, 0.50000000000000, 0.19230000000000],
    [0.38280000000000, 0.50000000000000, 0.80770000000000],
    [0.50000000000000, 0.19230000000000, 0.38280000000000],
    [0.50000000000000, 0.19230000000000, 0.61720000000000],
    [0.50000000000000, 0.80770000000000, 0.38280000000000],
    [0.50000000000000, 0.80770000000000, 0.61720000000000],
    [0.61720000000000, 0.50000000000000, 0.19230000000000],
    [0.61720000000000, 0.50000000000000, 0.80770000000000],
    [0.69230000000000, 0.00000000000000, 0.11720000000000],
    [0.69230000000000, 0.00000000000000, 0.88280000000000],
    [0.80770000000000, 0.38280000000000, 0.50000000000000],
    [0.80770000000000, 0.61720000000000, 0.50000000000000],
    [0.88280000000000, 0.30770000000000, 0.00000000000000],
    [0.88280000000000, 0.69230000000000, 0.00000000000000],
    [0.00000000000000, 0.25000000000000, 0.50000000000000],
    [0.00000000000000, 0.75000000000000, 0.50000000000000],
    [0.25000000000000, 0.50000000000000, 0.00000000000000],
    [0.50000000000000, 0.00000000000000, 0.25000000000000],
    [0.50000000000000, 0.00000000000000, 0.75000000000000],
    [0.75000000000000, 0.50000000000000, 0.00000000000000],
]
cell = [
    [10.35500000000000, 0.00000000000000, 0.00000000000000],
    [0.00000000000000, 10.35500000000000, 0.00000000000000],
    [0.00000000000000, 0.00000000000000, 10.35500000000000]
]
labels = ["Si"]*46
clathrate = Atoms(labels, scaled_positions=scaled_positions, cell=cell, pbc=True)

# Setup the symmetry analyzer
symm = SymmetryAnalyzer(clathrate, symmetry_tol=0.1)
has_free_param = symm.get_has_free_wyckoff_parameters()
wyckoff_sets_conv = symm.get_wyckoff_sets_conventional()

for i_group, group in enumerate(wyckoff_sets_conv):
    print("Set {}".format(i_group))
    print("    Letter: {}".format(group.wyckoff_letter))
    print("    Element: {}".format(group.element))
    print("    Indices: {}".format(group.indices))
    print("    Multiplicity: {}".format(group.multiplicity))
    print("    Repr.: {}".format(group.representative))
    print("    x: {}".format(group.x))
    print("    y: {}".format(group.y))
    print("    z: {}".format(group.z))
