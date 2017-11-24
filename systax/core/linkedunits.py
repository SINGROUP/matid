from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ase import Atoms


class LinkedUnitCollection(dict):
    """Represents a collection of similar cells that are connected in 3D space
    to form a structure, e.g. a surface.

    Essentially this is a special flavor of a regular dictionary: the keys can
    only be a sequence of three integers, and the values should be LinkedUnits.
    """
    def __init__(self, system):
        """
        Args:
            system(ase.Atoms): A reference to the system from which this
            LinkedUniCollection is gathered.
        """
        self.system = system
        dict.__init__(self)

    def __setitem__(self, key, value):
        # Transform key to tuple, check length
        try:
            key = tuple(key)
        except:
            raise TypeError(
                "Could not transform the given key '{}' into tuple."
                .format(key)
            )
        if len(key) != 3:
            raise ValueError(
                "The given coordinate '{}' does not have three components."
                .format(key)
            )

        # Check that old unit is not overwritten
        if key in dict.keys(self):
            raise ValueError(
                "Overriding existing units is not supported."
            )

        dict.__setitem__(self, key, value)

    def recreate_valid(self):
        """Used to recreate a new Atoms object, where each atom is created from
        a single unit cell. Atoms that were found not to belong to the periodic
        unit cell are not included.
        """
        recreated_system = Atoms(
            cell=self.system.get_cell(),
            pbc=self.system.get_pbc(),
        )
        for unit in self.values():
            i_all_indices = np.array(unit.all_indices)
            # print(i_all_indices)
            i_valid_indices = np.array([x for x in unit.basis_indices if x is not None])
            # print(i_valid_indices)
            i_atoms = self.system[i_all_indices[i_valid_indices]]
            recreated_system += i_atoms

        return recreated_system

    def get_indices(self):
        """Returns all the indices in of the LinedUnits in this collection as a
        single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        indices = []
        for unit in self.values():
            i_indices = unit.all_indices
            indices.extend(i_indices)

        return np.array(indices)

    # def get_layer_statistics(self):
        # """Returns all the indices in of the LinedUnits in this collection as a
        # single list.

        # Returns:
            # np.ndarray: Indices of the atoms in the original system that belong
            # to this collection of LinkedUnits.
        # """
        # indices = []
        # for unit in self.values():
            # i_indices = unit.all_indices
            # indices.extend(i_indices)

        # return np.array(indices)

    # def add_unit(self, new_unit, coordinate):

    # def get_surface_atoms(self):
        # pass

    # def get_sizes(self):
        # pass

    # def generate_units(self):

        # iterated_units = set()

        # # First return the unit at the origin
        # origin_coord = (0, 0, 0)
        # iterated_units.add(origin_coord)
        # yield(self.units[origin_coord])

        # # Recursively iterate over the neighbours
        # multipliers = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        # current_coord = np.array(origin_coord)
        # for multiplier in multipliers:
            # i_coord = current_coord + multiplier
            # i_unit = self.units[tuple(i_coord)]
            # yield i_unit

            # current_coord = i_unit

    # def generate_neighbors(self, unit):
        # coord = unit.index
        # multipliers = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        # for multiplier in

    # def __iter__(self):
        # return self

    # def __next__(self):
        # if self.current > self.high:
            # raise StopIteration
        # else:
            # self.current += 1
            # return self.current - 1


class LinkedUnit():
    """Represents a cell that is connected to others in 3D space to form a
    structure, e.g. a surface.
    """
    def __init__(self, index, seed_index, seed_coordinate, cell, all_indices, basis_indices):
        """
        Args:
            index(tuple of three ints):
            seed_index(int):
            seed_coordinate():
            cell(np.ndarray): Cell for this unit. Can change from unit to unit.
            all_indices(np.ndarray): Indices of all atoms in this unit
            basis_indices(sequence of ints and Nones): A sequence where there
                is an index or None for each atom that is supposed to be in the
                basis
        """
        self.index = index
        self.seed_index = seed_index
        self.seed_coordinate = seed_coordinate
        self.cell = cell
        self.all_indices = all_indices
        self.basis_indices = basis_indices
