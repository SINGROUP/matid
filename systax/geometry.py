"""This module defines functions for deriving geometry related quantities from
a atomic system.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ase.data import covalent_radii

from sklearn.cluster import DBSCAN


def get_moments_of_inertia(system, weight=True):
    """Calculates geometric inertia tensor, i.e., inertia tensor but with
    all masses are set to 1.

    I_ij = sum_k m_k (delta_ij * r_k^2 - x_ki * x_kj)
    with r_k^2 = x_k1^2 + x_k2^2 x_k3^2

    Args:
        system(ASE Atoms): Atomic system.

    Returns:
        (np.ndarray, np.ndarray): The eigenvalues and eigenvectors of the
        geometric inertia tensor.
    """
    # Move the origin to the geometric center
    positions = system.get_positions()
    centroid = get_center_of_mass(system, weight)
    pos_shifted = positions - centroid

    # Calculate the geometric inertia tensor
    if weight:
        weights = system.get_masses()
    else:
        weights = np.ones((len(system)))
    x = pos_shifted[:, 0]
    y = pos_shifted[:, 1]
    z = pos_shifted[:, 2]
    I11 = np.sum(weights*(y**2 + z**2))
    I22 = np.sum(weights*(x**2 + z**2))
    I33 = np.sum(weights*(x**2 + y**2))
    I12 = np.sum(-weights * x * y)
    I13 = np.sum(-weights * x * z)
    I23 = np.sum(-weights * y * z)

    I = np.array([
        [I11, I12, I13],
        [I12, I22, I23],
        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(I)

    return evals, evecs


def find_vacuum_directions(system, threshold=7.0):
    """Searches for vacuum gaps that are separating the periodic copies.

    TODO: Implement a n^2 search that allows the detection of more complex
    vacuum boundaries.

    Returns:
        np.ndarray: An array with a boolean for each lattice basis
        direction indicating if there is enough vacuum to separate the
        copies in that direction.
    """
    rel_pos = system.get_scaled_positions()
    pbc = system.get_pbc()

    # Find the maximum vacuum gap for all basis vectors
    gaps = np.empty(3, dtype=bool)
    for axis in range(3):
        if not pbc[axis]:
            gaps[axis] = True
            continue
        comp = rel_pos[:, axis]
        ind = np.sort(comp)
        ind_rolled = np.roll(ind, 1, axis=0)
        distances = ind - ind_rolled

        # The first distance is from first to last, so it needs to be
        # wrapped around
        distances[0] += 1

        # Find maximum gap in cartesian coordinates
        max_gap = np.max(distances)
        basis = system.get_cell()[axis, :]
        max_gap_cartesian = np.linalg.norm(max_gap*basis)
        has_vacuum_gap = max_gap_cartesian >= threshold
        gaps[axis] = has_vacuum_gap

    return gaps


def get_center_of_mass(system, weight=True):
    """
    """
    positions = system.get_positions()
    if weight:
        weights = system.get_masses()
    else:
        weights = np.ones((len(system)))
    cm = np.dot(weights, positions/weights.sum())

    return cm


def get_extended_system(system, target_size):
    """Replicate the system in different directions to reach a suitable
    system size for getting the moments of inertia.

    Args:
        system (ase.Atoms): The original system.
        target_size (float): The target size for the extended system.

    Returns:
        ase.Atoms: The extended system.
    """
    pbc = system.get_pbc()
    cell = system.get_cell()

    repetitions = np.array([1, 1, 1])
    for i, pbc in enumerate(pbc):
        # Only extend in the periodic dimensions
        basis = cell[i, :]
        if pbc:
            size = np.linalg.norm(basis)
            i_repetition = np.maximum(np.round(target_size/size), 1).astype(int)
            repetitions[i] = i_repetition

    extended_system = system.repeat(repetitions)

    return extended_system


def get_clusters(system, threshold=1.35):
    """
    """
    if len(system) == 1:
        return np.array([[0]])

    # Calculate distance matrix with radii taken into account
    distance_matrix = system.get_all_distances(mic=True)

    # Remove the radii from distances
    for i, i_number in enumerate(system.get_atomic_numbers()):
        for j, j_number in enumerate(system.get_atomic_numbers()):
            i_radii = covalent_radii[i_number]
            j_radii = covalent_radii[j_number]
            new_value = distance_matrix[i, j] - i_radii - j_radii
            distance_matrix[i, j] = max(new_value, 0)

    # Detect clusters
    db = DBSCAN(eps=threshold, min_samples=1, metric='precomputed', n_jobs=-1)
    db.fit(distance_matrix)
    clusters = db.labels_

    # Make a list of the different clusters
    idx_sort = np.argsort(clusters)
    sorted_records_array = clusters[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                    return_index=True)
    cluster_indices = np.split(idx_sort, idx_start[1:])

    return cluster_indices


def get_biggest_gap_indices(coordinates):
    """Given the list of coordinates for one axis, this function will find the
    maximum gap between them and return the index of the bottom and top
    coordinates. The bottom and top are defined as:

    ===       ===============    --->
        ^top    ^bot               ^axis direction
    """
    # Find the maximum vacuum gap for all basis vectors
    sorted_indices = np.argsort(coordinates)
    sorted_comp = coordinates[sorted_indices]
    rolled_comp = np.roll(sorted_comp, 1, axis=0)
    distances = sorted_comp - rolled_comp

    # The first distance is from first to last, so it needs to be
    # wrapped around
    distances[0] += 1

    # Find maximum gap
    bottom_index = sorted_indices[np.argmax(distances)]
    top_index = sorted_indices[np.argmax(distances)-1]

    return bottom_index, top_index


def get_dimensions(system, vacuum_gaps):
    """Given a system with vacuum gaps, calculate its dimensions in the
    directions with vacuum gaps by also taking into account the atomic radii.
    """
    orig_cell_lengths = np.linalg.norm(system.get_cell(), axis=1)

    # Create a repeated copy of the system. The repetition is needed in order
    # to get gaps to neighbouring cell copies in the periodic dimensions with a
    # vacuum gap
    sys = system.copy()
    sys = sys.repeat([2, 2, 2])

    dimensions = [None, None, None]
    numbers = sys.get_atomic_numbers()
    positions = sys.get_scaled_positions()
    radii = covalent_radii[numbers]
    cell_lengths = np.linalg.norm(sys.get_cell(), axis=1)
    radii_in_cell_basis = radii[:, None]/cell_lengths[None, :]

    for i_dim, vacuum_gap in enumerate(vacuum_gaps):
        if vacuum_gap:
            # Make a data structure containing the atom location information as
            # intervals from one side of the atom to the other in each
            # dimension.
            intervals = Intervals()
            for i_pos, pos in enumerate(positions[:, i_dim]):
                i_radii = radii_in_cell_basis[i_pos, i_dim]
                i_axis_start = pos - i_radii
                i_axis_end = pos + i_radii
                intervals.add_interval(i_axis_start, i_axis_end)

            # Calculate the maximum distance between atoms, when taking radius
            # into account
            gap = intervals.get_max_distance_between_intervals()
            gap = gap*cell_lengths[i_dim]
            dimensions[i_dim] = orig_cell_lengths[i_dim] - gap

    return dimensions


def get_wrapped_positions(scaled_pos, precision=1E-5):
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


def get_distance_matrix(cell, pos1, pos2, wrap_distances=False):
    """Calculates the distance matrix. If wrap_distances=True, calculates
    the matrix using periodic distances

    Args:
        cell():
        pos1():
        pos2():
    """
    if wrap_distances:
        rel_pos1 = to_scaled(cell, pos1)
        rel_pos2 = to_scaled(cell, pos2)
        disp_tensor = get_displacement_tensor(rel_pos1, rel_pos2)
        indices = np.where(disp_tensor > 0.5)
        disp_tensor[indices] = 1 - disp_tensor[indices]
        indices = np.where(disp_tensor < -0.5)
        disp_tensor[indices] = disp_tensor[indices] + 1
        disp_tensor = np.dot(disp_tensor, cell)
    else:
        disp_tensor = get_displacement_tensor(pos1, pos2)
    distance_matrix = np.linalg.norm(disp_tensor, axis=2)

    return distance_matrix


def get_displacement_tensor(pos1, pos2, pbc=None, cell=None):
    """Given an array of positions, calculates the 3D displacement tensor
    between the positions.

    The displacement tensor is a matrix where the entry A[i, j, :] is the
    vector pos1[i] - pos2[j], i.e. the vector from pos2 to pos1

    Args:
        pos1(np.ndarray): 2D array of positions
        pos2(np.ndarray): 2D array of positions
        pbc(boolean or a list of booleans): Periodicity of the axes
        cell(np.ndarray): Cell for taking into account the periodicity

    Returns:
        np.ndarray: 3D displacement tensor
    """
    if pbc is not None:
        pbc = expand_pbc(pbc)
        if pbc.any():
            if cell is None:
                raise ValueError(
                    "When using periodic boundary conditions you must provide "
                    "the cell."
                )

    # Make 1D into 2D
    shape1 = pos1.shape
    shape2 = pos2.shape
    if len(shape1) == 1:
        n_cols1 = len(pos1)
        pos1 = np.reshape(pos1, (-1, n_cols1))
    if len(shape2) == 1:
        n_cols2 = len(pos2)
        pos2 = np.reshape(pos2, (-1, n_cols2))

    # Add new axes so that broadcasting works nicely
    if pbc is None or not pbc.any():
        disp_tensor = pos1[:, None, :] - pos2[None, :, :]
    else:
        rel_pos1 = to_scaled(cell, pos1)
        rel_pos2 = to_scaled(cell, pos2)
        disp_tensor = rel_pos1[:, None, :] - rel_pos2[None, :, :]

        wrapped_disp_tensor = np.array(disp_tensor)
        for i, periodic in enumerate(pbc):
            if periodic:
                i_disp_tensor = disp_tensor[:, :, i]
                indices = np.where(i_disp_tensor > 0.5)
                i_disp_tensor[indices] = i_disp_tensor[indices] - 1
                indices = np.where(i_disp_tensor < -0.5)
                i_disp_tensor[indices] = i_disp_tensor[indices] + 1
                wrapped_disp_tensor[:, :, i] = i_disp_tensor
        disp_tensor = np.dot(wrapped_disp_tensor, cell)

    return disp_tensor


def expand_pbc(pbc):
    """Used to expand a pbc definition into a list of three booleans.

    Args:
        pbc(boolean or a list of booleans): The periodicity of the cell. This
            can be any of the values that is also supprted by ASE, namely: a
            boolean or a list of three booleans.

    Returns:
        np.ndarray of booleans: The periodicity expanded as an explicit list of
        three boolean values.
    """
    if pbc is True:
        new_pbc = [True, True, True]
    elif pbc is False:
        new_pbc = [False, False, False]
    elif len(pbc) == 3:
        new_pbc = pbc
    else:
        raise ValueError(
            "Could not interpret the given periodic boundary conditions: '{}'"
            .format(pbc)
        )

    return np.array(new_pbc)


def change_basis(positions, basis, offset=None):
    """Transform the given cartesian coordinates to a basis that is defined by
    the given basis and origin offset.

    Args:
        positions(np.ndarray): Positions in cartesian coordinates.
        basis(np.ndarray): Basis to which to transform.
        offset(np.ndarray): Offset of the origins. A vector from the old basis
            origin to the new basis origin.
    Returns:
        np.ndarray: Relative positions in the new basis
    """

    if offset is not None:
        positions -= offset
    new_basis_inverse = np.linalg.inv(basis.T)
    pos_prime = np.dot(positions, new_basis_inverse.T)

    return pos_prime


def get_positions_within_basis(system, basis, origin, tolerance, mask=[True, True, True]):
    """Used to return the indices of positions that are inside a certain basis.
    Also takes periodic boundaries into account.

    Args:
        system(ASE.Atoms): System from which the positions are searched.
        basis(np.ndarray): New basis vectors.
        origin(np.ndarray): New origin of the basis in cartesian coordinates.
        tolerance(float): The tolerance for the end points of the cell.
        mask(sequence of bool): Mask for selecting the basis's to consider.

    Returns:
        sequence of int: Indices of the atoms within this cell in the given
            system.
        np.ndarray: Relative positions of the found atoms.
    """
    # If the search extend beyound the cell boundary and periodic boundaries
    # allow, we must divide the search area into multiple regions

    # Transform positions into the new basis
    cart_pos = system.get_positions()

    # See if the new positions extend beyound the boundaries
    # rel_pos = system.get_scaled_positions()
    max_a = origin + basis[0, :]
    max_b = origin + basis[1, :]
    max_ab = origin + basis[0, :] + basis[1, :]
    cell = system.get_cell()
    rel_max_a = to_scaled(cell, max_a)
    rel_max_b = to_scaled(cell, max_b)
    rel_max_ab = to_scaled(cell, max_ab)

    directions = set()
    directions.add((0, 0, 0))
    for i_vec in [rel_max_a, rel_max_b, rel_max_ab]:

        a_max = i_vec[0]
        b_max = i_vec[1]

        if a_max >= 1 and b_max >= 1:
            directions.add((1, 1, 0))
        elif a_max >= 1 and 0 <= b_max < 1:
            directions.add((1, 0, 0))
        elif b_max >= 1 and 0 <= a_max < 1:
            directions.add((0, 1, 0))

    # If the new cell is overflowing beyound the boundaries of the original
    # system, we have to also check the periodic copies.
    indices = []
    a_prec, b_prec, c_prec = tolerance/np.linalg.norm(basis, axis=1)
    orig_basis = system.get_cell()
    cell_pos = []
    for i_dir in directions:

        i_origin = origin - np.dot(i_dir, orig_basis)
        vec_new = change_basis(cart_pos - i_origin, basis)

        # If no positions are defined, find the atoms within the cell
        for i_pos, pos in enumerate(vec_new):
            if mask[0]:
                x = 0 - a_prec <= pos[0] < 1 - a_prec
            else:
                x = True
            if mask[1]:
                y = 0 - b_prec <= pos[1] < 1 - b_prec
            else:
                y = True
            if mask[2]:
                z = 0 - c_prec <= pos[2] < 1 - c_prec
            else:
                z = True
            if x and y and z:
                indices.append(i_pos)
                cell_pos.append(pos)
    cell_pos = np.array(cell_pos)

    return indices, cell_pos


def get_matches(system, positions, numbers, tolerance):
    """Given a system and a list of cartesian positions and atomic numbers,
    returns a list of indices for the atoms corresponding to the given
    positions with some tolerance.

    Args:
        system(ASE.Atoms): System where to search the positions
        positions(np.ndarray): Positions to match in the system.
        tolerance(float): Maximum allowed distance that is required for a
            match in position.
    """
    orig_cart_pos = system.get_positions()
    orig_num = system.get_atomic_numbers()
    cell = system.get_cell()

    distances = get_distance_matrix(cell, positions, orig_cart_pos, wrap_distances=True)
    min_ind = np.argmin(distances, axis=1)
    matches = []
    for i, ind in enumerate(min_ind):
        distance = distances[i, ind]
        a_num = orig_num[ind]
        b_num = numbers[i]
        if distance <= tolerance and a_num == b_num:
            matches.append(ind)
        else:
            matches.append(None)

    return matches


def to_scaled(cell, positions, wrap=False, pbc=False):
    """Used to transform a set of positions to the basis defined by the
    cell of this system.

    Args:
        system(ASE.Atoms): Reference system.
        positions (numpy.ndarray): The positions to scale
        wrap (numpy.ndarray): Whether the positions should be wrapped
            inside the cell.

    Returns:
        numpy.ndarray: The scaled positions
    """
    pbc = expand_pbc(pbc)
    fractional = np.linalg.solve(
        cell.T,
        positions.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional[:, i] %= 1.0

    return fractional


def to_cartesian(cell, scaled_positions, wrap=False, pbc=False):
    """Used to transofrm a set of relative positions to the cartesian basis
    defined by the cell of this system.

    Args:
        system (ASE.Atoms): Reference system.
        positions (numpy.ndarray): The positions to scale
        wrap (numpy.ndarray): Whether the positions should be wrapped
            inside the cell.

    Returns:
        numpy.ndarray: The cartesian positions
    """
    pbc = expand_pbc(pbc)
    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                scaled_positions[:, i] %= 1.0

    cartesian_positions = np.dot(scaled_positions, cell)
    return cartesian_positions


def translate(system, translation, relative=False):
    """Translates the positions by the given translation.

    Args:
        translation (1x3 numpy.array): The translation to apply.
        relative (bool): True if given translation is relative to cell
            vectors.
    """
    if relative:
        rel_pos = system.get_scaled_positions()
        rel_pos += translation
        system.set_scaled_positions(rel_pos)
    else:
        cart_pos = system.get_positions()
        cart_pos += translation
        system.set_positions(cart_pos)


def get_surface_normal_direction(system):
    """Used to estimate a normal vector for a 2D like structure.

    Args:
        system (ase.Atoms): The system to examine.

    Returns:
        np.ndarray: The estimated surface normal vector
    """
    repeated = get_extended_system(system, 15)
    # vectors = system.get_cell()

    # Get the eigenvalues and eigenvectors of the moment of inertia tensor
    val, vec = get_moments_of_inertia(repeated)
    sorted_indices = np.argsort(val)
    val = val[sorted_indices]
    vec = vec[sorted_indices]

    # If the moment of inertia is not significantly bigger in one
    # direction, then the system cannot be described as a surface.
    moment_limit = 1.5
    if val[-1] < moment_limit*val[0] and val[-1] < moment_limit*val[1]:
        raise ValueError(
            "The given system could not be identified as a surface. Make"
            " sure that you provide a surface system with a sufficient"
            " vacuum gap between the layers (at least ~8 angstroms of vacuum"
            " between layers.)"
        )

    # The biggest component is the orhogonal one
    orthogonal_dir = vec[-1]

    return orthogonal_dir

    # Find out the cell direction that corresponds to the orthogonal one
    # cell = repeated.get_cell()
    # dots = np.abs(np.dot(orthogonal_dir, vectors.T))
    # orthogonal_vector_index = np.argmax(dots)
    # orthogonal_vector = vectors[orthogonal_vector_index]
    # orthogonal_dir = orthogonal_vector/np.linalg.norm(orthogonal_vector)

    return orthogonal_dir


def get_closest_direction(vec, directions, normalized=False):
    """Used to return the direction that is most parallel to a given one.

    Args:

    Returns:
    """
    if not normalized:
        directions = directions/np.linalg.norm(directions, axis=1)
    dots = np.abs(np.dot(vec, directions.T))
    index = np.argmax(dots)

    return index


# def get_displacement_tensor(self, system):
    # """A matrix where the entry A[i, j, :] is the vector
    # self.cartesian_pos[i] - self.cartesian_pos[j].

    # For periodic systems the distance of an atom from itself is the
    # smallest displacement of an atom from one of it's periodic copies, and
    # the distance of two different atoms is the distance of two closest
    # copies.

    # Returns:
        # np.array: 3D matrix containing the pairwise distance vectors.
    # """
    # if self.pbc.any():
        # pos = self.get_scaled_positions()
        # disp_tensor = pos[:, None, :] - pos[None, :, :]

        # # Take periodicity into account by wrapping coordinate elements
        # # that are bigger than 0.5 or smaller than -0.5
        # indices = np.where(disp_tensor > 0.5)
        # disp_tensor[indices] = 1 - disp_tensor[indices]
        # indices = np.where(disp_tensor < -0.5)
        # disp_tensor[indices] = disp_tensor[indices] + 1

        # # Transform to cartesian
        # disp_tensor = self.to_cartesian(disp_tensor)

        # # Figure out the smallest basis vector and set it as
        # # displacement for diagonal
        # cell = self.get_cell()
        # basis_lengths = np.linalg.norm(cell, axis=1)
        # min_index = np.argmin(basis_lengths)
        # min_basis = cell[min_index]
        # diag_indices = np.diag_indices(len(disp_tensor))
        # disp_tensor[diag_indices] = min_basis

    # else:
        # pos = self.get_positions()
        # disp_tensor = pos[:, None, :] - pos[None, :, :]

    # return disp_tensor


# def get_distance_matrix(self, system):
    # """Calculates the distance matrix A defined as:

        # A_ij = |r_i - r_j|

    # For periodic systems the distance of an atom from itself is the
    # smallest displacement of an atom from one of it's periodic copies, and
    # the distance of two different atoms is the distance of two closest
    # copies.

    # Returns:
        # np.array: Symmetric 2D matrix containing the pairwise distances.
    # """
    # displacement_tensor = self.get_displacement_tensor(system)
    # distance_matrix = np.linalg.norm(displacement_tensor, axis=2)
    # return distance_matrix


class Intervals(object):
    """Handles list of intervals.

    This class allows sorting and adding up of intervals and taking into
    account if they overlap.
    """
    def __init__(self, intervals=None):
        """Args:
            intervals: List of intervals that are added.
        """
        self._intervals = []
        self._merged_intervals = []
        self._merged_intervals_need_update = True
        if intervals is not None:
            self.add_intervals(intervals)

    def _add_up(self, intervals):
        """Add up the length of intervals.

        Argument:
            intervals: List of intervals that are added up.

        Returns:
            Result of addition.
        """
        if len(intervals) < 1:
            return None
        result = 0.
        for interval in intervals:
            result += abs(interval[1] - interval[0])
        return result

    def add_interval(self, a, b):
        """Add one interval.

        Args:
            a, b: Start and end of interval. The order does not matter.
        """
        self._intervals.append((min(a, b), max(a, b)))
        self._merged_intervals_need_update = True

    def add_intervals(self, intervals):
        """Add list of intervals.

        Args:
            intervals: List of intervals that are added.
        """
        for interval in intervals:
            if len(interval) == 2:
                self.add_interval(interval[0], interval[1])
            else:
                raise ValueError("Intervals must be tuples of length 2!")

    def set_intervals(self, intervals):
        """Set list of intervals.

        Args:
            intervals: List of intervals that are set.
        """
        self._intervals = []
        self.add_intervals(intervals)

    def remove_interval(self, i):
        """Remove one interval.

        Args:
            i: Index of interval that is removed.
        """
        try:
            del self._intervals[i]
            self._merged_intervals_need_update = True
        except IndexError:
            pass

    def get_intervals(self):
        """Returns the intervals.
        """
        return self._intervals

    def get_intervals_sorted_by_start(self):
        """Returns list with intervals ordered by their start.
        """
        return sorted(self._intervals, key=lambda x: x[0])

    def get_intervals_sorted_by_end(self):
        """Returns list with intervals ordered by their end.
        """
        return sorted(self._intervals, key=lambda x: x[1])

    def get_merged_intervals(self):
        """Returns list of merged intervals so that they do not overlap anymore.
        """
        if self._merged_intervals_need_update:
            if len(self._intervals) < 1:
                return self._intervals
            # sort intervals in list by their start
            sorted_by_start = self.get_intervals_sorted_by_start()
            # add first interval
            merged = [sorted_by_start[0]]
            # start from second interval
            for current in sorted_by_start[1:]:
                previous = merged[-1]
                # new interval if not current and previous are not overlapping
                if previous[1] < current[0]:
                    merged.append(current)
                # merge if current and previous are overlapping and if end of previous is expanded by end of current
                elif previous[1] < current[1]:
                    merged[-1] = (previous[0], current[1])
            self._merged_intervals = merged
            self._merged_intervals_need_update = False
        return self._merged_intervals

    def get_max_distance_between_intervals(self):
        """Returns the maximum distance between the intervals while accounting for overlap.
        """
        if len(self._intervals) < 2:
            return None
        merged_intervals = self.get_merged_intervals()
        distances = []
        if len(merged_intervals) == 1:
            return 0.0
        for i in range(len(merged_intervals) - 1):
            distances.append(abs(merged_intervals[i + 1][0] - merged_intervals[i][1]))
        return max(distances)

    def add_up_intervals(self):
        """Returns the added up lengths of intervals without accounting for overlap.
        """
        return self._add_up(self._intervals)

    def add_up_merged_intervals(self):
        """Returns the added up lengths of merged intervals in order to account for overlap.
        """
        return self._add_up(self.get_merged_intervals())
