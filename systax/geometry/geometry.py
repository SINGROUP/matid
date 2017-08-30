"""This module defines functions for deriving geometry related quantities from
a atomic system.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ase.data import covalent_radii

from sklearn.cluster import DBSCAN


def get_inertia_tensor(system, weight=True):
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

    val, vec = system.get_moments_of_inertia(vectors=True)
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


def get_clusters(system):
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
    db = DBSCAN(eps=1.3, min_samples=1, metric='precomputed', n_jobs=-1)
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
