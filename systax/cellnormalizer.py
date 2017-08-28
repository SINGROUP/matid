# import numpy as np

# from systax.exceptions import CellNormalizationError

# from systax.data.symmetry_data import \
    # TRANSLATIONS_DISCRETE
# from systax.data.alphabet_data import ALPHABET_POSITIONS
# from systax.core import System


# class CellNormalizer(object):

    # def normalize(
            # self,
            # symmetry_dataset,
            # orig_relative_pos,
            # orig_cell,
            # spglib_precision
            # ):
        # """Get the normalised system.

        # :param relative_pos: The original positions in the basis of the original
            # cell vectors.
        # :param symmetry_dataset: The symmetry dataset of the original system given
            # by spglib.
        # :param cell: The cell of the normalized system in angstroms.

        # :type relative_pos: numpy.array
        # :type symmetry_dataset: dict
        # :type cell_vectors: np.array

        # :return: The normalized system.
        # :rtype: :class:`System`

        # :raises: :class:`.CellNormalizationError`: If the normalized cell
            # coordinate cannot be succesfully transformed to the original cell
            # position.
        # """
        # if symmetry_dataset is None:
            # return None

        # space_group = symmetry_dataset["number"]
        # spglib_norm_lattice = np.array(symmetry_dataset["std_lattice"])
        # spglib_norm_pos = np.array(symmetry_dataset["std_positions"])
        # spglib_norm_numbers = symmetry_dataset["std_types"]
        # wyckoff_letters = symmetry_dataset["wyckoffs"]
        # equivalent_atoms = symmetry_dataset["equivalent_atoms"]
        # origin_shift = symmetry_dataset["origin_shift"]
        # transform = symmetry_dataset["transformation_matrix"]

        # # Get the Wyckoff letters of the atoms in the normalized lattice
        # try:
            # inverse = np.linalg.inv(transform)
        # except np.linalg.linalg.LinAlgError:
            # msg = (
                # "Error inverting the matrix that transforms positions from "
                # "original system to the spglib normalized cell."
                # )
            # raise CellNormalizationError(msg)
        # translated_shift = inverse.dot(origin_shift)

        # # Spglib precision is a diameter around a site, so we divide by two.
        # # Also it seems that spglib is not allowing the equal case, so only
        # # strictly smaller values are allowed.
        # allowed_offset = 0.5*spglib_precision

        # # For all atoms in the normalized cell, find the corresponding atom from
        # # the original cell and see which Wyckoff number is assigned to it
        # n_atoms = len(spglib_norm_numbers)
        # norm_wyckoff_letters = np.empty(n_atoms, dtype=str)
        # norm_equivalent_atoms = np.empty(n_atoms, dtype=int)

        # # Get the wrapped positions in the original cell
        # inverse_trans = inverse.T
        # translated_shift = np.dot(origin_shift, inverse_trans)
        # transformed_positions = np.dot(spglib_norm_pos, inverse_trans) - translated_shift
        # wrapped_positions = self.get_wrapped_positions(transformed_positions)

        # # for norm_coordinate in spglib_norm_pos:
        # for i_pos, wrapped_pos in enumerate(wrapped_positions):
            # index = self.search_periodic_positions(
                # wrapped_pos,
                # orig_relative_pos,
                # orig_cell,
                # allowed_offset
            # )
            # if index is None:
                # raise CellNormalizationError(
                    # "Could not find the corresponding atom for position {} in the "
                    # "original cell. Changing the precision might help."
                    # .format(wrapped_pos))
            # norm_wyckoff_letters[i_pos] = wyckoff_letters[index]
            # norm_equivalent_atoms[i_pos] = equivalent_atoms[index]

        # # Create the untranslated system
        # spglib_norm_system = System(
            # scaled_positions=spglib_norm_pos,
            # symbols=spglib_norm_numbers,
            # cell=spglib_norm_lattice,
            # wyckoff_letters=norm_wyckoff_letters,
            # equivalent_atoms=norm_equivalent_atoms
        # )

        # discrete_translations = TRANSLATIONS_DISCRETE.get(space_group)

        # # See if this space group has discrete translations available in some
        # # direction. Find a unique position among the translations.
        # discrete_translations = TRANSLATIONS_DISCRETE.get(space_group)
        # if discrete_translations:

            # # See if there is an alternative translational configuration with lower
            # # "score"
            # normalized_system = self.find_wyckoff_ground_state(
                # space_group,
                # spglib_norm_system,
                # discrete_translations
            # )
        # else:
            # normalized_system = spglib_norm_system

        # # Wrap everything inside the cell
        # normalized_system.wrap_positions()

        # return normalized_system

    # def get_wrapped_positions(self, positions, precision=1E-5, copy=True):
        # """Wrap the positions so that each element in the array is within the
        # half-closed interval [0, 1)

        # By wrapping values near 1 to 0 we will have a consistent way of
        # presenting systems.

        # :param positions: Atomic positions that are given in the unit cell basis.
        # :param precision: The precision for wrapping coordinates that are close to
            # zero or unity.
        # :param copy: Whether a the returned value is a copy or the values are
            # modified in-place.

        # :type positions: 2D numpy.array of float
        # :type precision: float
        # :type copy: bool

        # :return: The new wrapped positions.
        # :rtype: numpy.array
        # """
        # if copy:
            # wrapped_positions = np.copy(positions)
        # else:
            # wrapped_positions = positions

        # wrapped_positions %= 1
        # abs_zero = np.absolute(wrapped_positions)
        # abs_unity = np.absolute(abs_zero-1)

        # near_zero = np.where(abs_zero < precision)
        # near_unity = np.where(abs_unity < precision)

        # wrapped_positions[near_unity] = 0
        # wrapped_positions[near_zero] = 0

        # return wrapped_positions

    # def search_periodic_positions(
            # self,
            # target_pos,
            # positions,
            # cell,
            # accuracy):
        # """Searches a list of positions for a match for the target position taking
        # into account the periodicity of the system.

        # Args:
            # target_pos (1x3 np.array): The relative position to search.
            # positions (Nx3 np.array): The relative position where to search.
            # cell (3x3 np.array): The cell to find calculate a threshold accuracy in
                # cartesian coordinates.
            # accuracy (float): The minimum cartesian distance (angstroms) that is
                # required for the atoms to be considered identical.

        # Returns:
            # If a match is found, returns the index of the match in 'positions'. If
            # no match is found, returns None.
        # """
        # if len(positions.shape) == 1:
            # positions = positions[np.newaxis, :]

        # # Calculate the distances without taking into account periodicity.  Here we
        # # calculate all the distances although in reality we could loop over the
        # # distances and calculate only until the correct one is found. But it turns
        # # out it is faster to calculate the distances i one vectorized operation
        # # with numpy than a python loop.
        # displacements = positions-target_pos

        # # Take periodicity into account by wrapping coordinate elements that are
        # # bigger than 0.5 or smaller than -0.5
        # indices = np.where(displacements > 0.5)
        # displacements[indices] = 1 - displacements[indices]
        # indices = np.where(displacements < -0.5)
        # displacements[indices] = displacements[indices] + 1

        # # Convert displacements to cartesian coordinates
        # displacements = np.dot(displacements, cell.T)

        # # Try to find a match for the target by finding a distance that is less
        # # than the accuracy
        # distances = np.linalg.norm(displacements, axis=1)

        # min_index = np.argmin(distances)
        # min_distance = distances[min_index]
        # if min_distance <= accuracy:
            # return min_index
        # else:
            # return None

    # def find_wyckoff_ground_state(
            # self,
            # space_group,
            # system,
            # transform_list):
        # """For every available transformation (see the module
        # :mod:`Nomad.Tools.symmetryqueries.discrete_translations_query`), this
        # function will test and see which produces the state with the lowest score
        # defined by function :func:`calculate_wyckoff_score`. Returns the new
        # Wyckoff letters and atomic positions for this "ground state".

        # :param space_group: The space group of the cell
        # :param system: The original system with no translation applied.
        # :param translation_list: Translation objects for this space_group.

        # :type space_group: int
        # :type system: :class:`System`
        # :type transform_list: 3x4 ndarray

        # :return: The system corresponding to the configuration with the
            # lowest score.
        # :rtype: :class:`System`
        # """
        # # Calculate original score
        # original_score = self.calculate_wyckoff_score(
            # system.wyckoff_letters,
            # system.numbers
        # )

        # # Loop through available transforms, pick the one that produces lowest sum
        # # of wyckoff_letter*atomic_number
        # min_score = original_score
        # best_transform = None
        # best_wyckoff_letters = None
        # for transform in transform_list:
            # permutations = transform["permutations"]
            # new_wyckoff_letters = []
            # found = True
            # for wyckoff_letter in system.wyckoff_letters:
                # new_wyckoff = permutations.get(wyckoff_letter)
                # if new_wyckoff is None:
                    # found = False
                    # continue
                # else:
                    # new_wyckoff_letters.append(new_wyckoff)

            # # Calculate new score
            # if found:
                # new_score = self.calculate_wyckoff_score(
                    # new_wyckoff_letters,
                    # system.numbers
                # )
                # if new_score < min_score:
                    # min_score = new_score
                    # best_transform = transform["transformation"]
                    # best_wyckoff_letters = new_wyckoff_letters

        # if best_transform is None:
            # return system
        # else:
            # # Create the homogeneus coordinates
            # n_pos = len(system.relative_pos)
            # old_pos = np.empty((n_pos, 4))
            # old_pos[:, 3] = 1
            # old_pos[:, 0:3] = system.relative_pos

            # # Apply transformation with the augmented 3x4 matrix that is used for
            # # homogeneous coordinates
            # transformed_positions = np.dot(old_pos, best_transform.T)

            # new_system = System(
                # scaled_positions=transformed_positions,
                # symbols=system.numbers,
                # cell=system.get_cell(),
                # wyckoff_letters=np.array(best_wyckoff_letters),
                # equivalent_atoms=system.equivalent_atoms
            # )
            # new_system.wrap_positions()
            # return new_system

    # def calculate_wyckoff_score(self, wyckoff_letters, atomic_numbers):
        # """Calculate an internal score the given set of Wyckoff letters and atomic
        # numbers. This score is used to determine which representation is choosen
        # for the cell.

        # :param wyckoff_letters: The Wyckoff letters.
        # :param atomic_numbers: The atomic numbers (number of protons)

        # :type wyckoff_letters: 1D numpy.array of str
        # :type atomic_numbers: 1D numpy.array of float

        # :return: The score for the given setup.
        # :rtype: int
        # """
        # # Calculate score based on the multiplication of wyckoff letter position in
        # # alphabet and the atomic number
        # score = 0
        # for i_letter, letter in enumerate(wyckoff_letters):
            # alpha_factor = ALPHABET_POSITIONS[letter]
            # atomic_number = atomic_numbers[i_letter]
            # score += alpha_factor*atomic_number

        # return score
