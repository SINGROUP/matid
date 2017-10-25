"""
This script combines multiple pickle files into one python source file
containing the symmetry information.
"""
import os
import pickle
import pprint

_directory = os.path.dirname(os.path.abspath(__file__))

_improper_rigid_transformations_filename = os.path.join(_directory, "improper_rigid_transformations.pickle")
with open(_improper_rigid_transformations_filename, "rb") as fin:
    IMPROPER_RIGID_TRANSFORMATIONS = pickle.load(fin)

_proper_rigid_transformations_filename = os.path.join(_directory, "proper_rigid_transformations.pickle")
with open(_proper_rigid_transformations_filename, "rb") as fin:
    PROPER_RIGID_TRANSFORMATIONS = pickle.load(fin)

_space_group_info_filename = os.path.join(_directory, "space_group_info.pickle")
with open(_space_group_info_filename, "rb") as fin:
    SPACE_GROUP_INFO = pickle.load(fin)

# _translations_continuous_filename = os.path.join(_directory, "free_wyckoff_positions.pickle")
# with open(_translations_continuous_filename, "rb") as fin:
    # WYCKOFF_POSITIONS = pickle.load(fin)

with open("symmetry_data.py", "w") as fout:
    header = "from __future__ import absolute_import, division, print_function\n"
    header += "from numpy import array\n\n"
    fout.write(header)
    header_sgi = "SPACE_GROUP_INFO = "
    fout.write(header_sgi + pprint.pformat(SPACE_GROUP_INFO, indent=4) + "\n\n")
    # header_tc = "WYCKOFF_POSITIONS = "
    # fout.write(header_tc + pprint.pformat(WYCKOFF_POSITIONS, indent=4) + "\n\n")
    header_prop = "PROPER_RIGID_TRANSFORMATIONS = "
    fout.write(header_prop + pprint.pformat(PROPER_RIGID_TRANSFORMATIONS, indent=4) + "\n\n")
    header_improp = "IMPROPER_RIGID_TRANSFORMATIONS = "
    fout.write(header_improp + pprint.pformat(IMPROPER_RIGID_TRANSFORMATIONS, indent=4))
