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
            i_valid_indices = np.array([x for x in unit.basis_indices if x is not None])
            if len(i_valid_indices) != 0:
                i_atoms = self.system[i_valid_indices]
                recreated_system += i_atoms

        return recreated_system

    def get_basis_indices(self):
        """Returns the indices of the atoms that were found to belong to a unit
        cell basis in the LinkedUnits in this collection as a single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        indices = []
        for unit in self.values():
            i_indices = [x for x in unit.basis_indices if x is not None]
            indices.extend(i_indices)

        return np.array(indices)

    def get_inside_indices(self):
        """Returns the indices of the atoms that are within the region defined
        by the LinkedUnits in this collection as a single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        indices = []
        for unit in self.values():
            i_indices = unit.inside_indices
            indices.extend(i_indices)

        return np.array(indices)

    def get_indices(self):
        """Returns all the indices in of the LinkedUnits in this collection as a
        single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        basis_indices = set(self.get_basis_indices())
        inside_indices = (self.get_basis_indices())
        all_indices = basis_indices.union(inside_indices)

        return np.array(list(all_indices))


class LinkedUnit():
    """Represents a cell that is connected to others in 3D space to form a
    structure, e.g. a surface.
    """
    def __init__(self, index, seed_index, seed_coordinate, cell, basis_indices, inside_indices):
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
        self.basis_indices = basis_indices
        self.inside_indices = inside_indices
