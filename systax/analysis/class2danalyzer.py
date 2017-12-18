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

    # def get_thickness(self):
        # """Used to calculate the thickness of the structure. All 2D structures
        # should have a finite thickness.
        # """
        # gaps = self._get_vacuum_gaps()
        # if gaps.sum() != 1:
            # raise SystaxError(
                # "Found more than one dimension with a vacuum gap for a 2D "
                # "material."
            # )
        # orig_pos = self.system.get_positions()
        # gap_coordinates = orig_pos[:, gaps]
        # bottom_i, top_i = systax.geometry.get_biggest_gap_indices(gap_coordinates)

        # # Calculate height
        # bottom_pos = gap_coordinates[bottom_i]
        # top_pos = gap_coordinates[top_i]
        # height = top_pos - bottom_pos
        # if height < 0:
            # height += 1

        # return height.item()

    def is_pristine(self):
        """Looks at each unit cell in this system, and checks that each cell
        contains the same atoms in same positions. If so, then this material is
        labeled pristine.
        """
        # Check that there are no substitutions
        substitutions = self.unitcollection.get_substitutional_indices()
        if len(substitutions) != 0:
            return False

        basis_indices = set()

        # Check that no basis atom is shared with another cell
        for unit in self.unitcollection.values():
            basis = [x for x in unit.basis_indices if x is not None]
            intersection = basis_indices.intersection(basis)
            if len(intersection) != 0:
                return False
            basis_indices.update(basis)

        # Check that every layer has the same basis atoms present
        # TODO

        return True
