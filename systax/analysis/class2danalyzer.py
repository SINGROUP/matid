from __future__ import absolute_import, division, print_function, unicode_literals

import systax.geometry
from systax.exceptions import SystaxError
from systax.analysis.symmetryanalyzer import SymmetryAnalyzer
import numpy as np

from ase import Atoms
from ase.visualize import view

__metaclass__ = type


class Class2DAnalyzer(SymmetryAnalyzer):
    """Class for analyzing any 2D structure, like surfaces or 2D materials.
    """
    def __init__(self, system, vacuum_gaps, spglib_precision=None):
        """
        Args:
            system (ASE.Atoms): The system to inspect.
            spglib_precision (float): The tolerance for the symmetry detection
                done by spglib.
            vacuum_gaps: The directions in which there is a vacuum gap that
                separates periodic copies.
        """
        vacuum_gaps = np.array(vacuum_gaps)
        if vacuum_gaps.sum() != 1:
            raise SystaxError(
                "The given vacuum gaps do not contain exactly one direction in "
                "which there is a vacuum gap."
            )
        super().__init__(system, spglib_precision, vacuum_gaps)
        self.vacuum_index = np.where(vacuum_gaps == True)[0]

        # Break any possible symmetries in the nonperiodic direction
        # view(system)
        # centered_system = self.get_centered_system(system)
        # view(centered_system)
        # self.thickness = self.get_thickness(centered_system)
        # self.system = self.break_vacuum_symmetry(system, self.thickness)

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
        if self.thickness < 0.5*self.spglib_precision:
            is_flat = True

        # Find a proper rigid transformation that produces the best combination
        # of atomic species in the Wyckoff positions.
        space_group = self.get_space_group_number()
        wyckoff_letters, equivalent_atoms = \
            self._get_spglib_wyckoffs_and_equivalents_conventional()
        ideal_sys, ideal_wyckoff = self._find_wyckoff_ground_state(
            space_group,
            wyckoff_letters,
            spglib_conv_sys,
            is_flat=is_flat
        )

        # The idealization might move the system in the direction of the
        # nonperiodic axis. This centers the system back.
        centered_system = self.get_centered_system(ideal_sys)
        view(centered_system)

        # Minimize the cell to only just fit the atoms in the non-periodic
        # direction
        rel_pos = centered_system.get_scaled_positions()
        max_rel_pos = rel_pos[:, self.vacuum_index].max()
        new_cell = np.array(centered_system.get_cell())
        old_axis = np.array(new_cell[self.vacuum_index])
        new_axis = max_rel_pos*old_axis
        new_cell[self.vacuum_gaps, :] = new_axis
        centered_system.set_cell(new_cell)
        # view(centered_system)

        self._conventional_system = centered_system
        self._conventional_wyckoff_letters = ideal_wyckoff
        self._conventional_equivalent_atoms = equivalent_atoms
        return self._conventional_system

    def break_vacuum_symmetry(self, centered_system, thickness):
        """Used to ensure that the symmetries in the non-periodic direction is
        broken by adding a sufficient amount of vacuum in that direction.

        Args:
            system(ase.Atoms): The system in which the symmetries will be broken.
            thickness(float): The thickness of the structure in the direction
                in nonperiodic direction. Measured from center of topmost atom
                to center of lowermost atom.

        Returns:
            ase.Atoms: A new system in which the nonperiodic axis has been
            lengthened in order to break any possible symmetries.
        """
        axis = centered_system.get_cell()[self.vacuum_index]
        axis_length = np.linalg.norm(axis)
        axis_addition = max(2*thickness, 5)
        axis *= (1 + axis_addition/axis_length)
        new_system = centered_system.copy()
        new_cell = np.array(centered_system.get_cell())
        new_cell[self.vacuum_index] = axis
        new_system.set_cell(new_cell)

        return new_system

    def get_centered_system(self, system):
        """Used to center a 2D system so that the cell boundaries do not appear
        in the middle of the structure.
        """
        pos = system.get_positions()
        num = system.get_atomic_numbers()
        pbc = True
        cell = system.get_cell()

        # Calculate the displacements in the finite system taking into account periodicity
        displacements_finite_pbc = systax.geometry.get_displacement_tensor(pos, pos, cell, pbc, mic=True)

        # Bring the cluster together
        seed_pos = pos[0, :]
        disp_seed = displacements_finite_pbc[0, :, :]
        pos1 = seed_pos + disp_seed
        centered_system = Atoms(
            positions=pos1,
            cell=cell,
            symbols=num,
            pbc=pbc
        )

        # Wrap in periodic directions
        rel_pos = np.array(centered_system.get_scaled_positions(wrap=False))
        rel_pos[:, ~self.vacuum_gaps] %= 1

        # Set lowest position to zero
        min_pos = rel_pos[:, self.vacuum_index].min()
        rel_pos[:, self.vacuum_index] -= min_pos

        centered_system.set_scaled_positions(rel_pos)

        return centered_system

    def get_thickness(self, centered_system):
        """Used to calculate the thickness of the structure. All 2D structures
        should have a finite thickness.
        """
        pos_cart = centered_system.get_positions()
        gap_coordinates = pos_cart[:, self.vacuum_index]

        min_pos = gap_coordinates.min()
        max_pos = gap_coordinates.max()
        thickness = max_pos - min_pos

        return thickness
