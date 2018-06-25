from __future__ import absolute_import, division, print_function, unicode_literals

# The variable SYMMETRY_TOL controls the precision used by spglib in order
# to find symmetries. The atoms are allowed to move 1/2*SYMMETRY_TOL from
# their symmetry positions in order for spglib to still detect symmetries.
SYMMETRY_TOL = 2*0.2  # unit: angstrom

# The threshold for a system to be considered "flat". Used e.g. when
# determining if a 2D structure is purely 2-dimensional to allow extra rigid
# transformations that are improper in 3D but proper in 2D.
FLAT_DIM_THRESHOLD = 0.1

# An ordered list of Wyckoff letters
WYCKOFF_LETTERS = list("abcdefghijklmnopqrstuvwxyzA")
WYCKOFF_LETTER_POSITIONS = {letter: positions for positions, letter in enumerate(WYCKOFF_LETTERS)}

#===============================================================================
# Constants for classification
MAX_SINGLE_CELL_SIZE = 5
MAX_2D_CELL_HEIGHT = 5
# MAX_CELL_SIZE = [4, 12]
MAX_CELL_SIZE = [12]
# REL_POS_TOL = [0.25, 0.5, 0.75]  # Position tolerances. Given relative to minimum distance between two atoms in the system.
REL_POS_TOL = [0.25, 0.75]  # Position tolerances. Given relative to minimum distance between two atoms in the system.
# REL_POS_TOL = [0.25]  # Position tolerances. Given relative to minimum distance between two atoms in the system.
POS_TOL_SCALING = 0.0  # 0.0  # Per angstrom
ANGLE_TOL = 20
CLUSTER_THRESHOLD = 3.5
CRYSTALLINITY_THRESHOLD = 0.25
DELAUNAY_THRESHOLD = 1.5
BOND_THRESHOLD = 0.75
CHEM_SIMILARITY_THRESHOLD = 0.4
CELL_SIZE_TOL = 0.25
MAX_N_ATOMS = 1000
MIN_COVERAGE = 0.5
