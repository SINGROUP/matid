"""
This script is used to determine the transform from the system specififed by
first Hall number in each space group, into the system that correponds to the
conventional settings for that space group.

This information is needed because the normalizers and Wyckoff positions in
Bilbao Crystallographic Server are given in the conventional setttings only by
default.
"""
import spglib
from collections import defaultdict

space_hall_map = defaultdict(list)

for hall_number in range(1, 531):
    dataset = spglib.get_spacegroup_type(hall_number)
    number = dataset["number"]
    space_hall_map[number].append(hall_number)

degenerate_spgs = []
for key, value in space_hall_map.items():
    if len(value) == 1:
        continue

    degenerate_spgs.append(key)
    first_hall = value[0]
    dataset = spglib.get_spacegroup_type(first_hall)
    choice = dataset["choice"]

    # try:
        # origin = int(choice)
    # except ValueError as e:
        # if choice != "H":
            # print(choice)

    if choice == "":
        print(spglib.get_spacegroup_type(value[0])["choice"])
