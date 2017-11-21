from __future__ import absolute_import, division, print_function, unicode_literals

from systax.analysis.symmetryanalyzer import SymmetryAnalyzer
from systax.core.system import System

__metaclass__ = type


class Class3DAnalyzer(SymmetryAnalyzer):
    """Class for analyzing 3D structures.
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

        # Find a proper rigid transformation that produces the best combination
        # of atomic species in the Wyckoff positions.
        space_group = self.get_space_group_number()
        wyckoff_letters, equivalent_atoms = \
            self._get_spglib_wyckoffs_and_equivalents_conventional()
        ideal_sys, ideal_wyckoff = self._find_wyckoff_ground_state(
            space_group,
            wyckoff_letters,
            spglib_conv_sys
        )
        ideal_sys = System.from_atoms(ideal_sys)
        ideal_sys.set_equivalent_atoms(equivalent_atoms)
        ideal_sys.set_wyckoff_letters(ideal_wyckoff)

        self._conventional_system = ideal_sys
        self._conventional_wyckoff_letters = ideal_wyckoff
        self._conventional_equivalent_atoms = equivalent_atoms
        return ideal_sys
