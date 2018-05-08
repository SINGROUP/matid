"""
Goes through the spglib database for different Hall numbers and extracts space
group specific intormation. The results are then written to a python file for
later use.
"""
import spglib
import pickle

space_groups = {}
space_group_database = {}

for hall_number in range(1, 531):
    dataset = spglib.get_spacegroup_type(hall_number)
    number = dataset["number"]
    international_short = dataset["international_short"]

    # Check that the spglib data has no two different international symbols for
    # the same space group number
    old = space_groups.get(number)
    if old is not None:
        if old != international_short:
            raise LookupError("Two spacegroups have different point groups!")
    else:
        if number not in space_group_database:
            space_group_database[number] = {}

        # Point group. There actually seeems to be a bug in spglib 1.9.4, where
        # the Hermann-Mauguin point group symbol is in the plalce of Schonflies
        # data and vice versa.
        pointgroup = dataset["pointgroup_schoenflies"]
        space_group_database[number]["pointgroup"] = pointgroup

        # Crystal system
        crystal_systems = (
            ("triclinic", 1, 2),
            ("monoclinic", 3, 15),
            ("orthorhombic", 16, 74),
            ("tetragonal", 75, 142),
            ("trigonal", 143, 167),
            ("hexagonal", 168, 194),
            ("cubic", 195, 230),
        )
        crystal_system = None
        for system in crystal_systems:
            min_number = system[1]
            max_number = system[2]
            if number >= min_number and number <= max_number:
                crystal_system = system[0]
                break
        space_group_database[number]["crystal_system"] = crystal_system

        # Bravais lattice.
        lattice_type = international_short[0]
        crystal_system_map = {
            "triclinic": "a",
            "monoclinic": "m",
            "orthorhombic": "o",
            "tetragonal": "t",
            "trigonal": "h",
            "hexagonal": "h",
            "cubic": "c",
        }
        crystal_system_letter = crystal_system_map.get(crystal_system)
        space_group_database[number]["bravais_lattice"] = (
            crystal_system_letter +
            lattice_type
        )

# Write the results as a pickle file
with open("space_group_info.pickle", "wb") as fout:
    pickle.dump(space_group_database, fout)
