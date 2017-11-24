from __future__ import absolute_import, division, print_function, unicode_literals

import systax.geometry
from systax.exceptions import SystaxError
from systax.analysis.symmetryanalyzer import SymmetryAnalyzer
import numpy as np
import itertools

__metaclass__ = type


class Class2DAnalyzer(SymmetryAnalyzer):
    """Class for analyzing any 2D structure, like surfaces or 2D materials.
    """
    def get_conventional_system(self):
        """Returns an conventional description for this system. This
        description uses a conventional lattice where positions of the atoms,
        and the cell basis vectors are idealized to follow the symmetries that
        were found with the given precision. This means that e.g. the volume,
        angles between basis vectors and basis vector lengths may have
        deviations from the the original system.

        The following conventions are used to form the conventional system:

            - If there are multiple origins available for the space group, the
              one with the lowest Hall number is returned.

            - A proper rigid transformation that switches the atomic positions
              in the cell between different Wyckoff positions is applied to
              determine a unique representation for systems that do not have
              Wyckoff positions with free parameters.

            - The idealization of the lattice as defined by spglib, see
              https://atztogo.github.io/spglib/definition.html#conventions-of-standardized-unit-cell

        Returns:
            ASE.Atoms: The conventional system.
        """
        if self._conventional_system is not None:
            return self._conventional_system

        spglib_conv_sys = self._get_spglib_conventional_system()

        # Determine if the structure is flat. This will affect the
        # transformation that are allowed when finding the Wyckoff positions
        is_flat = False
        thickness = self.get_thickness()
        if thickness < 0.5*self.spglib_precision:
            is_flat = True

        # Find a proper rigid transformation that produces the best combination
        # of atomic species in the Wyckoff positions.
        space_group = self.get_space_group_number()
        wyckoff_letters, equivalent_atoms = \
            self._get_spglib_conventional_wyckoffs_and_equivalents()
        ideal_sys, ideal_wyckoff = self._find_wyckoff_ground_state(
            space_group,
            wyckoff_letters,
            spglib_conv_sys,
            is_flat=is_flat
        )

        self._conventional_system = ideal_sys
        self._conventional_wyckoff_letters = ideal_wyckoff
        self._conventional_equivalent_atoms = equivalent_atoms
        return ideal_sys

    def get_thickness(self):
        """Used to calculate the thickness of the structure. All 2D structures
        should have a finite thickness.
        """
        gaps = self._get_vacuum_gaps()
        if gaps.sum() != 1:
            raise SystaxError(
                "Found more than one dimension with a vacuum gap for a 2D "
                "material."
            )
        orig_pos = self.system.get_positions()
        gap_coordinates = orig_pos[:, gaps]
        bottom_i, top_i = systax.geometry.get_biggest_gap_indices(gap_coordinates)

        # Calculate height
        bottom_pos = gap_coordinates[bottom_i]
        top_pos = gap_coordinates[top_i]
        height = top_pos - bottom_pos
        if height < 0:
            height += 1

        return height.item()

    def get_layer_statistics(self):
        """Get statistics about the number of layers. Returns the average
        number of layers and the standard deviation in the number of layers.
        """
        # Find out which direction is the aperiodic one
        vacuum_gaps = self.vacuum_gaps
        vacuum_direction = self.system.get_cell()[vacuum_gaps]
        unit_cell = self.unitcollection[(0, 0, 0)].cell
        index = systax.geometry.get_closest_direction(vacuum_direction, unit_cell)
        mask = [True, True, True]
        mask[index] = False
        mask = np.array(mask)

        # Find out how many layers there are in the aperiodic directions
        max_sizes = {}
        min_sizes = {}
        for coord, unit in self.unitcollection.items():
            ab = tuple(np.array(coord)[mask])
            c = coord[index]
            min_size = min_sizes.get(ab)
            max_size = max_sizes.get(ab)
            if min_size is None or c < min_size:
                min_sizes[ab] = c
            if max_size is None or c > max_size:
                max_sizes[ab] = c

        sizes = []
        for key, max_c in max_sizes.items():
            min_c = min_sizes[key]
            size = max_c - min_c + 1
            sizes.append(size)
        sizes = np.array(sizes)

        mean = sizes.mean()
        std = sizes.std()

        return mean, std

    def is_pristine(self):
        """Looks at each unit cell in this system, and checks that each cell
        contains the same atoms in same positions. If so, then this material is
        labeled pristine.
        """
        # Check that is there a square-like area of unit cells
        # x_min = float("inf")
        # x_max = float("-inf")
        # y_min = float("inf")
        # y_max = float("-inf")
        # z_min = float("inf")
        # z_max = float("-inf")
        # for unit in self.unitcollection.values():
            # x, y, z = unit.index

            # if x < x_min:
                # x_min = x
            # elif x > x_max:
                # x_max = x

            # if y < y_min:
                # y_min = y
            # elif y > y_max:
                # y_max = y

            # if z < z_min:
                # z_min = z
            # elif z > z_max:
                # z_max = z

        # # Every index should be in the collection
        # indices = itertools.product(range(x_min, x_max+1), range(y_min, y_max+1), range(z_min, z_max+1))
        # for coord in indices:
            # unit = self.unitcollection.get(coord)
            # if unit is None:
                # print(coord)
                # print("bug")
                # return False

        # # Every coordinate in the collection should be within a rectangular
        # # region
        # for unit in self.unitcollection.values():
            # x, y, z = unit.index
            # x_ok = x_min < x < x_max
            # y_ok = y_min < y < y_max
            # z_ok = z_min < z < z_max

            # print(x_ok)
            # print(y_ok)
            # print(z_ok)

        # print(len(self.unitcollection))
        # Check that each unit contains only the basis
        for unit in self.unitcollection.values():
            basis_indices = unit.basis_indices
            for value in basis_indices:
                if value is None:
                    return False
        return True
