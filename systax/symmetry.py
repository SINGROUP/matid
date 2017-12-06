import numpy as np
from collections import OrderedDict, defaultdict
from systax.data.constants import WYCKOFF_LETTER_POSITIONS
import spglib


class WyckoffGroup():
    """Represents a group of atoms in a certain Wyckoff position, for a certain
    space group.
    """
    def __init__(self, wyckoff_letter, atomic_number, positions, space_group):
        self.wyckoff_letter = wyckoff_letter
        self.atomic_number = atomic_number
        self.positions = positions
        self.space_group = space_group
        self.x = None
        self.y = None
        self.z = None
        self.multiplicity = None


def make_wyckoff_groups(system, space_group):
    """Used to form a list of WyckoffGroup objects when given a space group and
    a system with wyckoff_letters and equivalent atoms.

    Args:
        system(System): The system for which the groups are formed. This system
        has to have the Wyckoff letters and the equivalent atoms set.
    Returns:
        OrderedDict: An ordered dictionary that maps a tuple (wyckoff letter,
            atomic number) into a list of WyckoffGroups. The order is such that
            they groups are first ordered by Wyckoff letter and then by the
            atomic number.
    """
    equivalent_atoms = system.get_equivalent_atoms()
    wyckoff_letters = system.get_wyckoff_letters()
    atomic_numbers = system.get_atomic_numbers()
    positions = system.get_scaled_positions()

    # Get a list of of unique elements in the list of equivalent atoms and also
    # store a mapping that can be used to construct the original array from the
    # unique elements
    uniques, indices = np.unique(equivalent_atoms, return_inverse=True)

    # Create a dictionary that maps each group to a set of of atoms
    groups = defaultdict(list)
    unique_to_indices = defaultdict(list)
    for i_index, index in enumerate(indices):
        uniq = uniques[index]
        unique_to_indices[uniq].append(i_index)

    # Create the WyckoffGroups
    for unique in uniques:

        i_positions = []
        for index in unique_to_indices[unique]:
            i_pos = positions[index]
            i_positions.append(i_pos)
        i_positions = np.array(i_positions)

        w = wyckoff_letters[index]
        z = atomic_numbers[index]

        group = WyckoffGroup(w, z, i_positions, space_group)
        groups[(w, z)].append(group)

    # Used for comparing by the position of the Wyckoff letter in a
    # predetermined list
    def compare(item1):
        return WYCKOFF_LETTER_POSITIONS[item1[0][0]]

    # First sort by Wyckoff letter, then by atomic number
    groups = OrderedDict(sorted(groups.items(), key=lambda t: (t[0][0], t[0][1])))

    return groups


def check_if_crystal(material3d_analyzer, threshold):
    """Quantifies the crystallinity of the structure as a ratio of symmetries
    per number of unique atoms in primitive cell. This metric can be used to
    distinguish between amorphous and 'regular' crystals.

    The number of symemtry operations corresponds to the symmetry operations
    corresponding to the hall number of the structure. The symmetry operations
    as given by spglib.get_symmetry() are specific to the original structure,
    and they have not been reduced to the symmetries of the space group.
    """
    # Get the number of equivalent atoms in the primitive cell.
    n_unique_atoms_prim = len(material3d_analyzer.get_equivalent_atoms_primitive())

    hall_number = material3d_analyzer.get_hall_number()
    sym_ops = spglib.get_symmetry_from_database(hall_number)
    n_symmetries = len(sym_ops["rotations"])

    ratio = n_symmetries/float(n_unique_atoms_prim)
    if ratio >= threshold:
        return True

    return False
