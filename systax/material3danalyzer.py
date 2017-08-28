import spglib

import numpy as np

from systax.utils.segfault_protect import segfault_protect
from systax.data.constants import SPGLIB_PRECISION, WYCKOFF_LETTERS
from systax.data.symmetry_data import PROPER_RIGID_TRANSFORMATIONS
from systax.exceptions import CellNormalizationError, SystaxError
from systax.core import System


class Material3DAnalyzer(object):
    """Class for analyzing 3D materials.
    """
    def __init__(self, system, spglib_precision=None):
        """
        Args:
            system (System): The system to inspect.
            spglib_precision (float): The tolerance for the symmetry detection
                done by spglib.
        """
        self.system = system
        if spglib_precision is None:
            self.spglib_precision = SPGLIB_PRECISION
        else:
            self.spglib_precision = spglib_precision

        self._symmetry_dataset = None
        self._conventional_system = None
        self._idealized_system = None
        self._spglib_idealized_system = None
        self._primitive_system = None

    def set_system(self, system):
        """Sets a new system for analysis.
        """
        self.reset()
        self.system = system

    def reset(self):
        """Used to reset all the cached values.
        """
        self._symmetry_dataset = None
        self._conventional_system = None
        self._primitive_system = None

    def get_spacegroup_number(self):
        """
        Returns:
            int: The spacegroup number.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["number"]

        return value

    def get_spacegroup_international_short(self):
        """
        Returns:
            str: The international spacegroup short symbol.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["international"]

        return value

    def get_hall_symbol(self):
        """
        Returns:
            str: The Hall symbol.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["hall"]

        return value

    def get_hall_number(self):
        """
        Returns:
            int: The Hall number.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["hall_number"]

        return value

    def get_point_group(self):
        """Symbol of the crystallographic point group in the Hermann–Mauguin
        notation.

        Returns:
            str: point group symbol
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["pointgroup"]

        return value

    def get_conventional_system(self):
        """Used to get the conventional representation of this system. This
        conventional cell represents a good approximation of how the given
        original system would look like in the conventional cell of the given
        material.


        Returns:
            System: The conventional system.
        """
        if self._conventional_system is not None:
            return self._conventional_system

        desc = self._system_to_spglib_description(self.system)
        conv_desc = spglib.standardize_cell(
            desc,
            to_primitive=False,
            no_idealize=True,
            symprec=self.spglib_precision)
        conv_sys = self._spglib_description_to_system(conv_desc)

        self._conventional_system = conv_sys
        return conv_sys

    def get_primitive_system(self):
        """Used to get the primitive representation of this system.

        Returns:
            System: The primitive system
        """
        if self._primitive_system is not None:
            return self._primitive_system

        desc = self._system_to_spglib_description(self.system)
        prim_desc = spglib.standardize_cell(
            desc,
            to_primitive=True,
            no_idealize=True,
            symprec=self.spglib_precision)
        prim_sys = self._spglib_description_to_system(prim_desc)

        self._primitive_system = prim_sys
        return prim_sys

    def get_idealized_system(self):
        """Returns an idealized description for this material. This idealized
        description uses a conventional lattice where positions of the atoms,
        and the cell basis vectors are idealized to follow the symmetries that
        were found with the given precision. This means that e.g. the volume,
        angles between basis vectors and basis vector lengths may have
        deviations from the the
        original system.

        The following conventions are used to form idealized system:

            - If there are multiple origins available for the space group, the
              one with the lowest Hall number is returned.

            - A proper rigid transformation that switches the atomic positions
              in the cell between different Wyckoff positions is applied to
              determine a unique representation for systems that do not have
              Wyckoff positions with free parameters.

            - The idealization of the lattice as defined by spglib, see
              https://atztogo.github.io/spglib/definition.html#conventions-of-standardized-unit-cell

        Returns:
            System: The idealized system.
        """
        if self._idealized_system is not None:
            return self._idealized_system

        spglib_ideal_sys = self._get_spglib_idealized_system()

        # Find a proper rigid transformation that produces the best combination
        # of atomic species in the Wyckoff positions.
        space_group = self.get_spacegroup_number()
        wyckoff_letters = self._get_spglib_wyckoff_letters()
        ideal_sys = self._find_wyckoff_ground_state(
            space_group,
            wyckoff_letters,
            spglib_ideal_sys
        )

        self._idealized_system = ideal_sys
        return ideal_sys

    def get_origin_shift(self):
        """The origin shift s that is needed to transform points in the
        original system to the conventional system. The relation between the
        original coordinates and the conventional coordinates is defined by:

        x' = P*x + s

        where x' is a coordinate in the conventional system, P is the
        transformation matrix, x is a coordinate in the original system and s
        is the origin shift.

        Returns:
            3*1 np.ndarray: The shift of the origin as a vector.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["origin_shift"]

        return value

    def get_transformation_matrix(self):
        """The transformation matrix P that transforms points in the original
        system to the conventional system. The relation between the original
        coordinates and the conventional coordinates is defined by:

        x' = P*x + s

        where x' is a coordinate in the conventional system, P is the
        transformation matrix, x is a coordinate in the original system and s
        is the origin shift.

        Returns:
            3x3 np.ndarray:
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["transformation_matrix"]

        return value

    def get_rotations(self):
        """Get the rotational parts of the Seits matrices that are associated
        with this space group. Each rotational matrix is accompanied by a
        translation with the same index.

        Returns:
            np.ndarray: Rotation matrices.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["rotations"]

        return value

    def get_translations(self):
        """Get the translational parts of the Seits matrices that are
        associated with this space group. Each translation is accompanied
        by a rotational matrix with the same index.

        Returns:
            np.ndarray: Translation vectors.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["translations"]

        return value

    def get_choice(self):
        """
        Returns:
            str: A string specifying the centring, origin and basis vector
            settings.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["choice"]

        return value

    def get_wyckoff_letters_original(self):
        """
        Returns:
            list of str: Wyckoff letters for the atoms in the original system.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["wyckoffs"]

        return value

    def get_wyckoff_letters_idealized(self):
        """
        Returns:
            list of str: Wyckoff letters for the atoms in the original system.
        """
        ideal_sys = self.get_idealized_system()
        letters = ideal_sys.get_wyckoff_letters()

        return letters

    def get_equivalent_atoms_original(self):
        """
        Returns:
            list of int: A list that maps each atom into a symmetry equivalent
                set.
        """
        dataset = self._get_symmetry_dataset()
        value = dataset["equivalent_atoms"]

        return value

    def _get_spglib_idealized_system(self):
        """Returns an idealized description for this material as defined by
        spglib.

        Returns:
            System: The idealized system as defined by spglib.
        """
        if self._spglib_idealized_system is not None:
            return self._spglib_idealized_system

        desc = self._system_to_spglib_description(self.system)
        ideal_desc = spglib.standardize_cell(
            desc,
            to_primitive=False,
            no_idealize=False,
            symprec=self.spglib_precision)
        ideal_sys = self._spglib_description_to_system(ideal_desc)

        self._spglib_idealized_system = ideal_sys
        return ideal_sys

    def _get_spglib_wyckoff_letters(self):
        """Return a list of Wyckoff letters for the atoms in the standardized
        cell defined by spglib. Note that these Wyckoff letters may not be the
        same as the ones given by get_idealized_system().

        Returns:
            list of str: List of Wyckoff letters for the atoms in the
            conventional system.
        """
        # conv_sys = self.get_conventional_system()
        conv_sys = self._get_spglib_idealized_system()
        conv_pos = conv_sys.get_scaled_positions()
        conv_num = conv_sys.get_atomic_numbers()

        orig_sys = self.system
        orig_pos = orig_sys.get_scaled_positions()
        orig_cell = orig_sys.get_cell()

        wyckoff_letters = self.get_wyckoff_letters_original()
        equivalent_atoms = self.get_equivalent_atoms_original()
        origin_shift = self.get_origin_shift()
        transform = self.get_transformation_matrix()

        # Get the Wyckoff letters of the atoms in the normalized lattice
        try:
            inverse = np.linalg.inv(transform)
        except np.linalg.linalg.LinAlgError:
            msg = (
                "Error inverting the matrix that transforms positions from "
                "original system to the spglib normalized cell."
                )
            raise SystaxError(msg)
        translated_shift = inverse.dot(origin_shift)

        # Spglib precision is a diameter around a site, so we divide by roughly two
        # to allow the "symmetrized" structure to match the original
        allowed_offset = 0.55*self.spglib_precision

        # For all atoms in the normalized cell, find the corresponding atom from
        # the original cell and see which Wyckoff number is assigned to it
        n_atoms = len(conv_num)
        norm_wyckoff_letters = np.empty(n_atoms, dtype=str)
        norm_equivalent_atoms = np.empty(n_atoms, dtype=int)

        # Get the wrapped positions in the original cell
        inverse_trans = inverse.T
        translated_shift = np.dot(origin_shift, inverse_trans)
        transformed_positions = np.dot(conv_pos, inverse_trans) - translated_shift
        wrapped_positions = self._wrap_positions(transformed_positions)

        for i_pos, wrapped_pos in enumerate(wrapped_positions):
            index = self._search_periodic_positions(
                wrapped_pos,
                orig_pos,
                orig_cell,
                allowed_offset
            )
            if index is None:
                print(wrapped_pos)
                print(orig_pos)
                raise SystaxError(
                    "Could not find the corresponding atom for position {} in the "
                    "original cell. Changing the precision might help."
                    .format(wrapped_pos))
            norm_wyckoff_letters[i_pos] = wyckoff_letters[index]
            norm_equivalent_atoms[i_pos] = equivalent_atoms[index]

        return norm_wyckoff_letters

    def _system_to_spglib_description(self, system):
        """Transforms the System object into a tuple used by spglib.
        """
        angstrom_cell = self.system.get_cell()
        relative_pos = self.system.get_scaled_positions()
        atomic_numbers = self.system.get_atomic_numbers()
        description = (angstrom_cell, relative_pos, atomic_numbers)

        return description

    def _spglib_description_to_system(self, desc):
        """Transforms the System object into a tuple used by spglib.
        """
        system = System(
            numbers=desc[2],
            cell=desc[0],
            scaled_positions=desc[1],
        )

        return system

    def _get_symmetry_dataset(self):
        """Calculates the symmetry dataset with spglib for the given system.
        """
        if self._symmetry_dataset is not None:
            return self._symmetry_dataset

        description = self._system_to_spglib_description(self.system)
        # Spglib has been observed to cause segmentation faults when fed with
        # invalid data, so run in separate process to catch those cases
        try:
            symmetry_dataset = segfault_protect(
                spglib.get_symmetry_dataset,
                description,
                self.spglib_precision)
        except RuntimeError:
            raise CellNormalizationError(
                "Segfault in spglib when finding symmetry dataset. Please check "
                " the given cell, scaled positions and atomic numbers."
            )
        if symmetry_dataset is None:
            raise CellNormalizationError(
                'Spglib error when finding symmetry dataset.')

        self._symmetry_dataset = symmetry_dataset

        return symmetry_dataset

    def _wrap_positions(self, positions, precision=1E-5, copy=True):
        """Wrap positions so that each element in the array is within the
        half-closed interval [0, 1)

        By wrapping values near 1 to 0 we will have a consistent way of
        presenting systems.

        Args:
            positions (np.ndarray): Atomic positions that are given in the unit cell basis.
            precision (float): The precision for wrapping coordinates that are close to

                zero or unity.
            copy (bool): Whether a the returned value is a copy or the values are
                modified in-place.

        Returns:
            np.ndarray: The new wrapped positions.
        """
        if copy:
            wrapped_positions = np.copy(positions)
        else:
            wrapped_positions = positions

        wrapped_positions %= 1
        abs_zero = np.absolute(wrapped_positions)
        abs_unity = np.absolute(abs_zero-1)

        near_zero = np.where(abs_zero < precision)
        near_unity = np.where(abs_unity < precision)

        wrapped_positions[near_unity] = 0
        wrapped_positions[near_zero] = 0

        return wrapped_positions

    def _search_periodic_positions(
            self,
            target_pos,
            positions,
            cell,
            accuracy):
        """Searches a list of positions for a match for the target position taking
        into account the periodicity of the system.

        Args:
            target_pos (1x3 np.array): The relative position to search.
            positions (Nx3 np.array): The relative position where to search.
            cell (3x3 np.array): The cell to find calculate a threshold accuracy in
                cartesian coordinates.
            accuracy (float): The minimum cartesian distance (angstroms) that is
                required for the atoms to be considered identical.

        Returns:
            If a match is found, returns the index of the match in 'positions'. If
            no match is found, returns None.
        """
        if len(positions.shape) == 1:
            positions = positions[np.newaxis, :]

        # Calculate the distances without taking into account periodicity.
        # Here we calculate all the distances although in reality we could loop
        # over the distances and calculate only until the correct one is found.
        # But it turns out it is faster to calculate the distances i one
        # vectorized operation with numpy than a python loop.
        displacements = positions-target_pos

        # Take periodicity into account by wrapping coordinate elements that
        # are bigger than 0.5 or smaller than -0.5
        indices = np.where(displacements > 0.5)
        displacements[indices] = 1 - displacements[indices]
        indices = np.where(displacements < -0.5)
        displacements[indices] = displacements[indices] + 1

        # Convert displacements to cartesian coordinates
        displacements = np.dot(displacements, cell.T)

        # Try to find a match for the target by finding a distance that is less
        # than the accuracy
        distances = np.linalg.norm(displacements, axis=1)

        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        if min_distance <= accuracy:
            return min_index
        else:
            return None

    def _find_wyckoff_ground_state(
            self,
            space_group,
            old_wyckoff_letters,
            system):
        """
        """
        transform_list = PROPER_RIGID_TRANSFORMATIONS[space_group]

        # Form a mapping between transformation number and a list of wyckoff
        # letters for the atoms. The transformation with numberr -1 corresponds
        # to the original system
        systems = {-1: old_wyckoff_letters}
        # systems = {}
        for i_transform, transform in enumerate(transform_list):
            permutations = transform["permutations"]
            new_wyckoff_letters = []
            found = False

            for old_wyckoff_letter in old_wyckoff_letters:
                new_wyckoff = permutations.get(old_wyckoff_letter)
                if new_wyckoff is not None:
                    found = True
                    new_wyckoff_letters.append(new_wyckoff)
                else:
                    new_wyckoff_letters.append(old_wyckoff_letter)

            if found:
                systems[i_transform] = new_wyckoff_letters

        # For each abailable transform, determine a mapping between a Wyckoff
        # letter and a list of atomic numbers with that Wyckoff letter
        atomic_numbers = system.get_atomic_numbers()
        mappings = {}
        for i_transform, new_wyckoff in systems.items():
            i_wyckoff_to_number_map = {}
            for wyckoff_letter in WYCKOFF_LETTERS:
                i_atomic_numbers = []
                for i_atom, i_letter in enumerate(new_wyckoff):
                    if wyckoff_letter == i_letter:
                        i_atomic_number = atomic_numbers[i_atom]
                        i_atomic_numbers.append(i_atomic_number)
                if len(i_atomic_numbers) is not 0:
                    i_wyckoff_to_number_map[wyckoff_letter] = np.array(sorted(i_atomic_numbers))
            mappings[i_transform] = i_wyckoff_to_number_map

        # Find which transformation produces the combination of wyckoff letters
        # and atomic numbers that is most favourable
        best_transform_i = None
        for letter in WYCKOFF_LETTERS:

            # First find out the systems with this letter
            numbers = []
            indices = []
            for i_system, i_mapping in mappings.items():
                i_numbers = i_mapping.get(letter)
                if i_numbers is not None:
                    numbers.append(i_numbers)
                    indices.append(i_system)

            # If only one system found with this letter, then it is the best as
            # the Wyckoff letters are enumerated in a predetermined order.
            if len(numbers) == 1:
                best_transform_i = indices[0]
                break

            # Next try to see if one of the systems has lower atomic numbers
            # with this letter
            found = False
            numbers = np.array(numbers)
            for i_col in range(numbers.shape[1]):
                col = numbers[:, i_col]
                col_min_ind = np.where(col == col.min())
                if len(col_min_ind) > 1:
                    numbers = numbers[col_min_ind]
                else:
                    best_transform_i = col_min_ind[0]
                    break
                    found = True
            if found:
                break

        # Apply the best transform
        new_system = system.copy()
        if best_transform_i == -1:
            new_system.set_wyckoff_letters(old_wyckoff_letters)
            return new_system
        else:
            best_transform = transform_list[best_transform_i]["transformation"]

            # Create the homogeneus coordinates
            n_pos = len(system)
            old_pos = np.empty((n_pos, 4))
            old_pos[:, 3] = 1
            old_pos[:, 0:3] = system.get_scaled_positions()

            # Apply transformation with the augmented 3x4 matrix that is used for
            # homogeneous coordinates
            transformed_positions = np.dot(old_pos, best_transform.T)

            # Get rid of the extra dimension from homogeneous coordinates
            transformed_positions = transformed_positions[:, 0:3]

            # Wrap the positions to the half-closed interval [0, 1)
            wrapped_pos = self._get_wrapped_positions(transformed_positions)
            new_system.set_scaled_positions(wrapped_pos)

            return new_system

    def _get_wrapped_positions(self, scaled_pos, precision=1E-5):
        """Wrap the given relative positions so that each element in the array
        is within the half-closed interval [0, 1)

        By wrapping values near 1 to 0 we will have a consistent way of
        presenting systems.
        """
        scaled_pos %= 1

        abs_zero = np.absolute(scaled_pos)
        abs_unity = np.absolute(abs_zero-1)

        near_zero = np.where(abs_zero < precision)
        near_unity = np.where(abs_unity < precision)

        scaled_pos[near_unity] = 0
        scaled_pos[near_zero] = 0

        return scaled_pos
