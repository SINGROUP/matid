from __future__ import absolute_import, division, print_function, unicode_literals

# The size limit for analysing a system. Systems bigger than this will not be
# analyzed.
SYSTEM_SIZE_LIMIT = 100

# The variable SPGLIB_PRECISION controls the precision used by spglib in order
# to find symmetries. The atoms are allowed to move 1/2*SPGLIB_PRECISION from
# their symmetry positions in order for spglib to still detect symmetries.
SPGLIB_PRECISION = 2*0.2  # unit: angstrom

# Defines the "bin size" for rounding cell angles for the material hash
ANGLE_ROUNDING = float(10.0)  # unit: degree

# The threshold for point equality in k-space
K_SPACE_PRECISION = 150e6  # unit 1/m

# The energy threshold for how much a band can be on top or below the fermi
# level in order to detect a gap
FERMI_LEVEL_PRECISION = 300*1.38064852E-23  # k_B x T at room temperature, unit: Joule

# The threshold for a system to be considered "flat". Used e.g. when
# determining if a 2D structure is purely 2-dimensional to allow extra rigid
# transformations that are improper in 3D but proper in 2D.
FLAT_DIM_THRESHOLD = 0.1

# An ordered list of Wyckoff letters
WYCKOFF_LETTERS = list("abcdefghijklmnopqrstuvwxyzA")
WYCKOFF_LETTER_POSITIONS = {letter: positions for positions, letter in enumerate(WYCKOFF_LETTERS)}
