"""
This script combines multiple pickle files into one python source file
containing the symmetry information.
"""
import os
import pickle
import pprint
import numpy as np


def print_dict(d, level):
    out = []
    out.append("{\n")
    for key in sorted(d.keys()):
        value = d[key]
        if isinstance(value, np.ndarray):
            value = "array(" + np.array2string(value, threshold=np.inf, max_line_width=np.inf, separator=',', sign=" ", prefix="", suffix="").replace("\n", "") + ")"
        elif isinstance(value, dict):
            value = print_dict(value, level+1)
        else:
            value = repr(value)
        if isinstance(key, str):
            key = "\""+key+"\""
        out.append("    " * level + "{}: {},\n".format(key, value))
    out.append("    " * (level-1) + "}")
    return "".join(out)


_directory = os.path.dirname(os.path.abspath(__file__))

CHIRALITY_PRESERVING_EUCLIDEAN_NORMALIZERS = {}
batches = [0, 100, 200, 230]
for i in range(len(batches) - 1):
    start = batches[i] + 1
    end = batches[i + 1]
    with open(f"chirality_preserving_euclidean_normalizers_{start}_{end}.pickle", "rb") as fin:
        CHIRALITY_PRESERVING_EUCLIDEAN_NORMALIZERS.update(pickle.load(fin))

# _space_group_info_filename = os.path.join(_directory, "space_group_info.pickle")
# with open(_space_group_info_filename, "rb") as fin:
    # SPACE_GROUP_INFO = pickle.load(fin)

# _translations_continuous_filename = os.path.join(_directory, "free_wyckoff_positions.pickle")
# with open(_translations_continuous_filename, "rb") as fin:
    # WYCKOFF_POSITIONS = pickle.load(fin)

# wyckoff_sets_path = os.path.join(_directory, "wyckoff_sets.pickle")
# with open(wyckoff_sets_path, "rb") as fin:
#     wyckoff_sets = pickle.load(fin)

with open("symmetry_data_new.py", "w") as fout:
    # header += "from numpy import array\n\n"
    # fout.write(header)
    # header_sgi = "SPACE_GROUP_INFO = "
    # fout.write(header_sgi + pprint.pformat(SPACE_GROUP_INFO, indent=4) + "\n\n")
    # header_tc = "WYCKOFF_POSITIONS = "
    # fout.write(header_tc + pprint.pformat(WYCKOFF_POSITIONS, indent=4) + "\n\n")
    header_norm = "CHIRALITY_PRESERVING_EUCLIDEAN_NORMALIZERS = "
    fout.write(header_norm + pprint.pformat(CHIRALITY_PRESERVING_EUCLIDEAN_NORMALIZERS, indent=4) + "\n\n")
    # header_wyckoff_sets = "WYCKOFF_SETS = "
    # fout.write(header_wyckoff_sets + print_dict(wyckoff_sets, 1))

