from __future__ import absolute_import, division, print_function, unicode_literals

import spglib

import numpy as np

from systax.utils.segfault_protect import segfault_protect
from systax.data.constants import SPGLIB_PRECISION, WYCKOFF_LETTERS
from systax.data.symmetry_data import PROPER_RIGID_TRANSFORMATIONS, IMPROPER_RIGID_TRANSFORMATIONS
from systax.exceptions import CellNormalizationError, SystaxError
import systax.geometry

from ase import Atoms

from fractions import Fraction

import sys

import abc

__metaclass__ = type

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Analyzer(ABC):
    """A base class for all classes that are used to analyze structures.
    """
    def __init__(self, system, spglib_precision=None, vacuum_gaps=None):
        """
        Args:
            system (ASE.Atoms): The system to inspect.
            spglib_precision (float): The tolerance for the symmetry detection
                done by spglib.
        """
        self.system = system
        self.vacuum_gaps = vacuum_gaps
        if spglib_precision is None:
            self.spglib_precision = SPGLIB_PRECISION
        else:
            self.spglib_precision = spglib_precision

        self.reset()

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
        self._spglib_conventional_system = None
        self._spglib_wyckoff_letters = None
        self._spglib_equivalent_atoms = None
        self._conventional_wyckoff_letters = None
        self._conventional_equivalent_atoms = None
        self._best_transform = None

    def get_space_group_number(self):
        """
        Returns:
            int: The space group number.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["number"]

        return value

    def get_space_group_international_short(self):
        """
        Returns:
            str: The international space group short symbol.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["international"]

        return value

    def get_hall_symbol(self):
        """
        Returns:
            str: The Hall symbol.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["hall"]

        return value

    def get_hall_number(self):
        """
        Returns:
            int: The Hall number.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["hall_number"]

        return value

    def get_point_group(self):
        """Symbol of the crystallographic point group in the Hermann-Mauguin
        notation.

        Returns:
            str: point group symbol
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["pointgroup"]

        return value

    def get_primitive_system(self):
        """Returns an primitive description for this system. This description
        uses a primitive lattice where positions of the atoms, and the cell
        basis vectors are idealized to follow the symmetries that were found
        with the given precision. This means that e.g. the volume, angles
        between basis vectors and basis vector lengths may have deviations from
        the the original system.

        Returns:
            ASE.Atoms: The primitive system.
        """
        if self._primitive_system is not None:
            return self._primitive_system

        conv_sys = self.get_conventional_system()
        centring = self.get_space_group_international_short()[0]

        # For the primitive centering the conventional lattice is the primitive
        # as well
        if centring == "P":
            self._primitive_system = conv_sys
            return conv_sys

        primitive_transformations = {
            "A": np.array([
                [1, 0, 0],
                [0, 1/2, -1/2],
                [0, 1/2, 1/2],
            ]),
            "C": np.array([
                [1/2, 1/2, 0],
                [-1/2, 1/2, 0],
                [0, 0, 1],
            ]),
            "R": np.array([
                [2/3, -1/3, -1/3],
                [1/3, 1/3, -2/3],
                [1/3, 1/3, 1/3],
            ]),
            "I": np.array([
                [-1/2, 1/2, 1/2],
                [1/2, -1/2, 1/2],
                [1/2, 1/2, -1/2],
            ]),
            "F": np.array([
                [0, 1/2, 1/2],
                [1/2, 0, 1/2],
                [1/2, 1/2, 0],
            ]),
        }

        # Transform conventional cell to the primitive cell
        transform = primitive_transformations[centring]
        conv_cell = conv_sys.get_cell()
        prim_cell = np.dot(transform.T, conv_cell)

        # Transform all position to the basis of the primitive cell
        conv_pos = conv_sys.get_positions()
        prim_cell_inv = np.linalg.inv(prim_cell)
        prim_pos = np.dot(conv_pos, prim_cell_inv)

        # See which positions are inside the cell in the half-closed interval
        conv_num = conv_sys.get_atomic_numbers()
        inside_mask = np.all((prim_pos >= 0) & (prim_pos < 1-1e-8), axis=1)
        inside_pos = prim_pos[inside_mask]
        inside_num = conv_num[inside_mask]

        prim_sys = Atoms(
            scaled_positions=inside_pos,
            symbols=inside_num,
            cell=prim_cell
        )

        self._primitive_system = prim_sys
        return prim_sys

        # if self._primitive_system is not None:
            # return self._primitive_system

        # desc = self._system_to_spglib_description(self.system)
        # prim_desc = spglib.standardize_cell(
            # desc,
            # to_primitive=True,
            # no_idealize=False,
            # symprec=self.spglib_precision)
        # prim_sys = self._spglib_description_to_system(prim_desc)

        # self._primitive_system = prim_sys
        # return prim_sys

    @abc.abstractmethod
    def get_conventional_system(self):
        """Used to get the conventional representation of this system.
        """

    def get_conventional_lattice_fit(self):
        """Used to get a 3x3 matrix representing a fit of the original
        simulation cell to the conventional cell. This lattice can be e.g. used
        to calculate the lattice parameters or the volume of the original
        system in the lattice of the conventional system. The order of the cell
        basis vectors is the same as in the cell returned by
        get_conventional_system().get_cell().

        Returns:
            np.ndarray: The lattice basis vectors as a matrix.
        """
        conv_sys = self.get_conventional_system()
        conv_cell = self.get_conventional_system().get_cell()
        orig_sys = self.system
        orig_cell = orig_sys.get_cell()
        orig_cell_inv = np.linalg.inv(orig_cell.T)
        coeff = np.dot(conv_cell, orig_cell_inv)

        # Round the coefficients to a reasonable fractional number
        for i in range(0, coeff.shape[0]):
            for j in range(0, coeff.shape[0]):
                old_value = coeff[i, j]
                new_value = Fraction(old_value).limit_denominator(10)
                coeff[i, j] = new_value

        # Remake the conventional basis vectors in the new scaled coordinates
        new_std_lattice = np.dot(coeff, orig_cell)

        # Ensure that the volume is preserved by scaling the lattice
        # appropriately
        vol_original = orig_sys.get_volume()
        vol_conv = abs(np.linalg.det(new_std_lattice))
        n_atoms_original = len(orig_sys)
        n_atoms_conv = len(conv_sys)

        vol_per_atom_orig = vol_original/n_atoms_original
        vol_per_atom_conv = vol_conv/n_atoms_conv

        factor = vol_per_atom_orig/vol_per_atom_conv
        new_std_lattice *= factor

        return new_std_lattice

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
        dataset = self.get_symmetry_dataset()
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
        dataset = self.get_symmetry_dataset()
        value = dataset["transformation_matrix"]

        return value

    def get_rotations(self):
        """Get the rotational parts of the Seits matrices that are associated
        with this space group. Each rotational matrix is accompanied by a
        translation with the same index.

        Returns:
            np.ndarray: Rotation matrices.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["rotations"]

        return value

    def get_translations(self):
        """Get the translational parts of the Seits matrices that are
        associated with this space group. Each translation is accompanied
        by a rotational matrix with the same index.

        Returns:
            np.ndarray: Translation vectors.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["translations"]

        return value

    def get_choice(self):
        """
        Returns:
            str: A string specifying the centring, origin and basis vector
            settings.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["choice"]

        return value

    def get_wyckoff_letters_original(self):
        """
        Returns:
            list of str: Wyckoff letters for the atoms in the original system.
        """
        spglib_wyckoffs = self._get_spglib_wyckoff_letters_original()
        if self._best_transform is None:
            self.get_conventional_system()
        permutations = self._best_transform["permutations"]
        new_wyckoffs = []
        for old_wyckoff in spglib_wyckoffs:
            new_wyckoff = permutations[old_wyckoff]
            new_wyckoffs.append(new_wyckoff)

        return new_wyckoffs

    def get_equivalent_atoms_original(self):
        """
        Returns:
            list of int: A list that maps each atom into a symmetry equivalent
                set.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["equivalent_atoms"]

        return value

    def get_wyckoff_letters_conventional(self):
        """Get the Wyckoff letters of the atoms in the conventional system.

        Returns:
            list of str: Wyckoff letters.
        """
        if self._conventional_wyckoff_letters is None:
            self.get_conventional_system()
        return self._conventional_wyckoff_letters

    def get_equivalent_atoms_conventional(self):
        """List of equivalent atoms in the idealized system.

        Returns:
            list of int: A list that maps each atom into a symmetry equivalent
                set.
        """
        if self._spglib_equivalent_atoms is None:
            self._get_spglib_wyckoff_letters_and_equivalent_atoms()
        return self._spglib_equivalent_atoms

    def _get_spglib_conventional_system(self):
        """Returns an idealized description for this material as defined by
        spglib.

        Returns:
            ASE.Atoms: The idealized system as defined by spglib.
        """
        if self._spglib_conventional_system is not None:
            return self._spglib_conventional_system

        dataset = self.get_symmetry_dataset()
        cell = dataset["std_lattice"]
        pos = dataset["std_positions"]
        num = dataset["std_types"]
        spg_conv_sys = self._spglib_description_to_system((cell, pos, num))

        self._spglib_conventional_system = spg_conv_sys
        return spg_conv_sys

    def _get_spglib_wyckoff_letters_original(self):
        """
        Returns:
            list of str: Wyckoff letters for the atoms in the original system.
        """
        dataset = self.get_symmetry_dataset()
        value = dataset["wyckoffs"]

        return value

    def _get_spglib_conventional_wyckoffs_and_equivalents(self):
        """Return a list of Wyckoff letters for the atoms in the standardized
        cell defined by spglib. Note that these Wyckoff letters may not be the
        same as the ones given by get_idealized_system().

        Returns:
            list of str: List of Wyckoff letters for the atoms in the
            conventional system.
        """
        if self._spglib_wyckoff_letters is not None and \
           self._spglib_equivalent_atoms is not None:
            return self._spglib_wyckoff_letters, self._spglib_equivalent_atoms

        conv_sys = self._get_spglib_conventional_system()
        conv_pos = conv_sys.get_scaled_positions()
        conv_num = conv_sys.get_atomic_numbers()

        orig_sys = self.system
        orig_pos = orig_sys.get_scaled_positions()
        orig_cell = orig_sys.get_cell()

        wyckoff_letters = self._get_spglib_wyckoff_letters_original()
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
                raise SystaxError(
                    "Could not find the corresponding atom for position {} in the "
                    "original cell. Changing the precision might help."
                    .format(wrapped_pos))
            norm_wyckoff_letters[i_pos] = wyckoff_letters[index]
            norm_equivalent_atoms[i_pos] = equivalent_atoms[index]

        self._spglib_wyckoff_letters = norm_wyckoff_letters
        self._spglib_equivalent_atoms = norm_equivalent_atoms

        return norm_wyckoff_letters, norm_equivalent_atoms

    def _system_to_spglib_description(self, system):
        """Transforms the given ASE.Atoms object into a tuple used by spglib.
        """
        angstrom_cell = self.system.get_cell()
        relative_pos = self.system.get_scaled_positions()
        atomic_numbers = self.system.get_atomic_numbers()
        description = (angstrom_cell, relative_pos, atomic_numbers)

        return description

    def _spglib_description_to_system(self, desc):
        """Transforms a tuple used by spglib into ASE.Atoms
        """
        system = Atoms(
            numbers=desc[2],
            cell=desc[0],
            scaled_positions=desc[1],
        )

        return system

    def get_symmetry_dataset(self):
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
            system,
            is_flat=False):
        """
        """

        # Gether the allowed transformations. For completely flat structures (all
        # atoms in 2D plane), also the rigid transformation that are improper
        # (determinant -1), will become proper, as we can always invert the
        # non-periodic axis to change the sign of the determinant. Physically this
        # corresponds to rotating the system rigidly through the third nonperiodic
        # dimension.
        transform_list = []
        identity = {
            "transformation": np.identity(4),
            "permutations": {x: x for x in old_wyckoff_letters},
        }
        transform_list.append(identity)

        proper_rigid_trans = PROPER_RIGID_TRANSFORMATIONS.get(space_group)
        if proper_rigid_trans is not None:
            transform_list.extend(proper_rigid_trans)
        if is_flat:
            improper_rigid_trans = IMPROPER_RIGID_TRANSFORMATIONS.get(space_group)
            if improper_rigid_trans is not None:
                transform_list.extend(improper_rigid_trans)

        # If no transformation found for this space group, return the same
        # system
        if len(transform_list) == 1:
            self._best_transform = identity
            return system, old_wyckoff_letters

        # print(old_wyckoff_letters)
        # print(system.get_chemical_symbols())
        # print(transform_list)

        # Form a mapping between transformation number and a list of wyckoff
        # letters for the atoms. The transformation with numberr -1 corresponds
        # to the original system
        transformed_wyckoffs = {}
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
                transformed_wyckoffs[i_transform] = new_wyckoff_letters

        # For each available transform, determine a mapping between a Wyckoff
        # letter and a list of atomic numbers with that Wyckoff letter
        atomic_numbers = system.get_atomic_numbers()
        mappings = {}
        for i_transform, new_wyckoff in transformed_wyckoffs.items():
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

        searched_indices = mappings.keys()
        for letter in WYCKOFF_LETTERS:

            # print(letter)

            # First find out the systems with this letter
            numbers = []
            i_indices = []
            for index in searched_indices:
                i_mapping = mappings[index]
                i_numbers = i_mapping.get(letter)
                if i_numbers is not None:
                    numbers.append(i_numbers)
                    i_indices.append(index)

            i_indices = np.array(i_indices)
            numbers = np.array(numbers)

            # If only one transformation found with this letter, then that
            # transformation is the best
            if len(numbers) == 1:
                best_transform_i = i_indices[0]
                break
            # If no transformation with this letter found, move onto the next
            # letter
            elif len(numbers) == 0:
                continue

            # Next try to see if one of the systems has atoms with lower atomic
            # numbers with this letter
            found = False
            for i_col in range(numbers.shape[1]):
                col = numbers[:, i_col]
                col_min_ind = np.where(col == col.min())[0]
                if len(col_min_ind) > 1:
                    numbers = numbers[col_min_ind]
                else:
                    best_transform_i = i_indices[col_min_ind[0]]
                    found = True
                    break
            if found:
                break
            else:
                searched_indices = i_indices[col_min_ind]

        # If no best transformation was found, then multiple transformation are
        # equal. Ensure this and then return one of them.
        if best_transform_i is None:
            for i in range(len(searched_indices)-1):
                index1 = searched_indices[i]
                index2 = searched_indices[i+1]
                wyckoffs1 = transformed_wyckoffs[index1]
                wyckoffs2 = transformed_wyckoffs[index2]
                if wyckoffs1 != wyckoffs2:
                    raise SystaxError("Could not successfully decide best Wyckoff positions.")
            best_transform_i = searched_indices[0]

        # Apply the best transform
        new_system = system.copy()
        if best_transform_i == 0:
            self._best_transform = identity
            return new_system, old_wyckoff_letters
        else:
            self._best_transform = transform_list[best_transform_i]
            transformation_matrix = self._best_transform["transformation"]

            # Create the homogeneus coordinates
            n_pos = len(system)
            old_pos = np.empty((n_pos, 4))
            old_pos[:, 3] = 1
            old_pos[:, 0:3] = system.get_scaled_positions()

            # Apply transformation with the augmented 3x4 matrix that is used
            # for homogeneous coordinates
            transformed_positions = np.dot(old_pos, transformation_matrix.T)

            # Get rid of the extra dimension of the homogeneous coordinates
            transformed_positions = transformed_positions[:, 0:3]

            # Wrap the positions to the half-closed interval [0, 1)
            wrapped_pos = systax.geometry.get_wrapped_positions(transformed_positions)
            new_system.set_scaled_positions(wrapped_pos)

            return new_system, transformed_wyckoffs[best_transform_i]

    def _get_vacuum_gaps(self, threshold=7.0):
        # if self.vacuum_gaps is not None:
            # return self.vacuum_gaps

        gaps = systax.geometry.find_vacuum_directions(self.system, threshold=threshold)
        return gaps
