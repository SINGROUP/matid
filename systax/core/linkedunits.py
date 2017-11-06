from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


class LinkedUnitCollection():
    """Represents a collection of similar cells that are connected in 3D space
    to form a structure, e.g. a surface.
    """
    def __init__(self, system):
        self.units = {}
        self.system = system

    def add_unit(self, new_unit, coordinate):
        self.units[tuple(coordinate)] = new_unit

    def get_surface_atoms(self):
        pass

    def get_sizes(self):
        pass

    # def generate_units(self):
        # # First return the unit at the origin
        # yield(units[(0, 0, 0)])

        # # Recursively iterate over the neighbours
        # for

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
    def __init__(self, index, seed_index, seed_coordinate, cell, indices):
        self.neighbors = {}
        self.index = index
        self.seed_index = seed_index
        self.seed_coordinate = seed_coordinate
        self.cell = cell
        self.indices = indices

    def get_neighbor(self, coord):
        value = self.neighbors[coord]
        return value

    def set_neighbor(self, coord, value):
        self.neighbors[coord] = value
