"""This module defines functions for deriving geometry related quantities from
a atomic system.
"""
import math
import itertools
from collections import defaultdict

import numpy as np
from numpy.random import RandomState

from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii
from ase import Atom, Atoms
import ase.geometry

from matid.data.element_data import get_covalent_radii
from matid.core.linkedunits import Substitution
from matid.core.distances import Distances
import matid.geometry

from sklearn.cluster import DBSCAN

from scipy.spatial import Delaunay
import spglib


def get_nearest_atom(system, position, mic=True):
    """Finds the index of the atom nearest to the given position in the given
    system.

    Args:
        system(ase.Atoms): The system from which the atom is searched from.
        position(np.ndarray): The position to search.
        mic(boolean): Whether to use the minimum image convention on periodic
            systems.

    Returns:
        int: Index of the nearest atom in the given system.
    """
    positions = system.get_positions()
    pbc = system.get_pbc()
    cell = system.get_cell()
    distances = get_distance_matrix(position, positions, cell=cell, pbc=pbc, mic=mic)
    min_index = np.argmin(distances)

    return min_index


def get_dimensionality(
        system,
        cluster_threshold,
        dist_matrix_radii_mic_1x=None,
        return_clusters=False,
        radii="covalent",
    ):
    """Used to calculate the dimensionality of a system with a modified
    Topology Scaling Algorithm (TSA) (Michael Ashton, Joshua Paul, Susan B.
    Sinnott, and Richard G. Hennig Phys. Rev. Lett. 118, 106101).

    Args:
        system (ASE.Atoms): The system for which the dimensionality is
            evaluated.
        cluster_threshold(float): The epsilon value for the DBSCAN algorithm
            that is used to identify clusters within the unit cell.
        dist_matrix_radii_pbc (np.ndarray): A precalculated distance matrix
            that takes in to account the periodicity and has the covalent radii of
            the atoms already subtracted.
        return_clusters(boolean): Whether the clusters are returned
        radii(str|np.ndarray): The radii to use for atoms. Use either a preset
            or a custom list or atomic radii where the atomic number is used as an
            index. The available presets are:

                - covalent: Covalent radii from DOI:10.1039/B801115J
                - vdw: van Der Waals radii from DOI:10.1039/C3DT50599E
                - vdw_covalent: preferably van Der Waals radii, covalent if vdw
                  not defined.

    Returns:
        int|none: The dimensionality of the system. If the dimensionality can't be
            evaluated because the system has multiple disconnected components in
            the original cell, None is returned.
        list: A list of clusters. Each entry in the list contains the indices
            of atoms in a cluster.
    """
    system_1x = system
    pbc = system_1x.get_pbc()
    num_1x = system_1x.get_atomic_numbers()
    cell_1x = system_1x.get_cell()

    # When calculating the displacement tensor we can ignore neighbouring
    # copies that are farther away than the clustering cutoff. The maximum mic
    # distance to allow is cluster_threshold + 2*max_radii
    if isinstance(radii, str):
        if radii == "covalent":
            radii = covalent_radii
        elif radii == "vdw":
            radii = vdw_radii
        elif radii == "vdw_covalent":
            radii = np.array([vdw_radii[i] if vdw_radii[i] != np.nan else covalent_radii[i] for i in range(len(vdw_radii))])
    system_radii = radii[num_1x]
    max_radii = system_radii.max()
    max_distance = cluster_threshold + 2*max_radii

    # 1x1x1 system
    if dist_matrix_radii_mic_1x is None:
        pos_1x = system.get_positions()
        displacements_mic_1x, dist_matrix_mic_1x = get_displacement_tensor(
            pos_1x,
            pos_1x,
            cell_1x,
            pbc,
            mic=True,
            max_distance=max_distance,
            return_distances=True
        )
        radii_1x = radii[num_1x]
        radii_matrix_1x = radii_1x[:, None] + radii_1x[None, :]
        dist_matrix_radii_mic_1x = dist_matrix_mic_1x - radii_matrix_1x

    # Check the number of clusters.
    clusters_1x = get_clusters(dist_matrix_radii_mic_1x, cluster_threshold, min_samples=1)
    n_clusters_1x = len(clusters_1x)

    # If the system consists of multiple components that are not connected
    # according to the clustering done here, then we cannot assess the
    # dimensionality.
    if n_clusters_1x > 1:
        dim = None
    else:
        # 2x2x2 system
        n_pbc = np.sum(pbc)
        if n_pbc > 0:

            repeats = np.array([1, 1, 1])
            repeats[pbc] = 2
            system_2x = system.repeat(repeats)
            pos_2x = system_2x.get_positions()
            cell_2x = system_2x.get_cell()
            num_2x = system_2x.get_atomic_numbers()
            displacements_mic_2x, dist_matrix_mic_2x = get_displacement_tensor(
                pos_2x,
                pos_2x,
                cell_2x,
                pbc,
                mic=True,
                max_distance=max_distance,
                return_distances=True
            )
            radii_2x = radii[num_2x]
            radii_matrix_2x = radii_2x[:, None] + radii_2x[None, :]
            dist_matrix_radii_mic_2x = dist_matrix_mic_2x - radii_matrix_2x

            # Check the number of clusters.
            clusters_2x = get_clusters(dist_matrix_radii_mic_2x, cluster_threshold, min_samples=1)
            n_clusters_2x = len(clusters_2x)

            # This is the analytic formula for the dimensionality based on cluster
            # size on a 2x supersystem.
            dim = int(n_pbc - math.log(n_clusters_2x, 2))
        else:
            dim = 0

    if return_clusters:
        return dim, clusters_1x
    else:
        return dim


def get_tetrahedra_decomposition(system, max_distance):
    """Used to decompose a series of 3D atomic coordinates into non-overlapping
    tetrahedron that together represent the atomic structure.

    """
    # TODO: Reuse the distance matrix from the original system, just filter out
    # the entries that do not belong to the valid basis. This way we can avoid
    # recalculating the distances.

    # TODO: Determine the distance matrix of the padded system during the
    # construction of the padded system. This should be doable by simply using
    # the internal displacements and the cell periodicity.

    class TetrahedraDecomposition():
        """A class that represents a collection of tetrahedron.
        """
        def __init__(self, delaunay, invalid_simplex_indices):
            self.delaunay = delaunay
            self.invalid_simplex_indices = invalid_simplex_indices

        def find_simplex(self, pos):
            """Used to find the index of the simplex in which the given
            position resides in.

            Args:
                pos(np.nadrray): Position for which to find the simplex

            Returns:
                int: Index of the simplex in which the point is in. Returns
                None if simplex was not found.
            """
            index = self.delaunay.find_simplex(pos).item()
            if index == -1:
                return None
            if index not in self.invalid_simplex_indices:
                return index
            return None

    cell = system.get_cell()
    pos = system.get_positions()
    num = system.get_atomic_numbers()

    # Calculate a matrix containing the combined radii for each pair of atoms
    # in an extended system
    radii = covalent_radii[num]
    radii_matrix = radii[:, None] + radii[None, :]

    displacements_finite = get_displacement_tensor(pos, pos)

    # In order for the decomposition to cover also the edges, we have to extend
    # the system to cover also into the adjacent periodic images. That is done
    # within this loop.
    tesselation_atoms = system.copy()
    multipliers = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

    for mult in multipliers:
        if tuple(mult) != (0, 0, 0):
            disloc = np.dot(mult, cell)

            # If system is periodic in this direction, calculate the distance
            # between atoms in the periodically repeated images and choose
            # atoms from the copies that are within a certain range when tha
            # radii are taken into account
            disp = np.array(displacements_finite)
            disp += disloc
            dist = np.linalg.norm(disp, axis=2)
            dist_radii = dist - radii_matrix
            connectivity_mask = np.where(dist_radii < max_distance)

            pad_pos = pos[connectivity_mask[1]] + disp[connectivity_mask[0], connectivity_mask[1]]
            pad_num = num[connectivity_mask[1]]
            pad_atoms = Atoms(positions=pad_pos, symbols=pad_num)
            tesselation_atoms += pad_atoms

    # view(tesselation_atoms)
    tesselation_pos = tesselation_atoms.get_positions()

    # The QJ options makes sure that all positions are included in the
    # tesselation
    tri = Delaunay(tesselation_pos, qhull_options="QJ")

    # Remove invalid simplices from the surface until only valid simplices are
    # found on the surface
    distance_matrix = get_covalent_distances(tesselation_atoms, mic=False)
    simplices = np.array(tri.simplices)
    neighbors = np.array(tri.neighbors)
    end = False
    invalid_indices = set()
    surface_simplices = np.any(neighbors == -1, axis=1)
    original_indices = np.arange(len(simplices))
    surface_simplex_indices = set(original_indices[surface_simplices])

    while not end:

        # Remove the surface simplices that are too big. Also update the
        # neighbour list.
        too_big_simplex_indices = []
        for i_simplex in surface_simplex_indices:
            simplex = simplices[i_simplex]
            for i, j in itertools.combinations(simplex, 2):
                distance = distance_matrix[i, j]
                if distance >= max_distance:
                    too_big_simplex_indices.append(i_simplex)
                    break
        invalid_indices = invalid_indices | set(too_big_simplex_indices)
        if len(too_big_simplex_indices) == 0:
            end = True
        else:
            # Find the neighbours of the simplices that were removed
            mask = np.isin(neighbors, too_big_simplex_indices)
            affected = np.any(mask, axis=1)
            new_surface_simplex_indices = set(original_indices[affected])
            new_surface_simplex_indices -= invalid_indices
            surface_simplex_indices = new_surface_simplex_indices

    tetrahedras = TetrahedraDecomposition(tri, invalid_indices)
    return tetrahedras


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


def get_center_of_mass(system):
    """Calculates the center of mass and also takes the periodicity of the
    system into account.

    The algorithm is replicated from:
    https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions

    Args:
        system(ase.Atoms): The system for which the center of mass is
            calculated.

    Returns:
        np.ndarray: The cartesian positions of the center of mass in the given
        system.
    """
    pbc = system.get_pbc()
    relative_positions = system.get_scaled_positions()
    masses = system.get_masses()
    total_mass = np.sum(masses)
    cell = system.get_cell()

    rel_com = np.zeros((3))
    for i_comp in range(3):
        i_pbc = pbc[i_comp]
        i_pos = relative_positions[:, i_comp]
        if i_pbc:
            theta = i_pos * 2 * np.pi
            xi = np.cos(theta) * masses
            zeta = np.sin(theta) * masses

            xi_mean = np.mean(xi)
            zeta_mean = np.mean(zeta)

            mean_theta = np.arctan2(-zeta_mean, -xi_mean) + np.pi
            com_rel = mean_theta / (2 * np.pi)
            rel_com[i_comp] = com_rel
        else:
            rel_com[i_comp] = np.sum(i_pos * masses) / total_mass

    com_cart = to_cartesian(cell, rel_com)[0, :]

    return com_cart


def get_space_filling(system):
    """Calculates the ratio of vacuum to filled space by assuming covalent
    radii for the atoms.

    Args:
        system(ASE.Atoms): Atomic system.

    Returns:
        float: The ratio of occupied volume to the cell volume.
    """
    cell_volume = system.get_volume()
    atomic_numbers = system.get_atomic_numbers()
    occupied_volume = 0
    radii = get_covalent_radii(atomic_numbers)
    volumes = 4.0/3.0*np.pi*radii**3
    occupied_volume = np.sum(volumes)
    ratio = occupied_volume/cell_volume

    return ratio


def make_random_displacement(system, delta, rng=None):
    """Dislocate every atom in the given system in a random direction but by
    the same amount. Uses an internal random number generator to avoid touching
    the global numpy.random.seed()-function.

    Args:
        system(ASE.Atoms): The system for which the displacement are performed.
        delta(float): The magnitude of the displacements.
        rng(np.random.RandomState): Random number generator.
    """
    if rng is None:
        rng = RandomState()
    pos = system.get_positions()
    n_atoms = len(system)
    disloc = rng.rand(n_atoms, 3)
    disloc /= np.linalg.norm(disloc, axis=1)[:, None]
    disloc *= delta
    new_pos = pos + disloc
    system.set_positions(new_pos)


def get_extended_system(system, target_size):
    """Replicate the system in different directions to reach the given target
    size.

    Args:
        system (ase.Atoms): The original system.
        target_size (float): The target size for the extended system in
            angstroms.

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


def get_clusters(dist_matrix, threshold, min_samples=1):
    """Used to detect clusters with the DBSCAN algorithm.

    Args:
        dist_matrix(np.ndarray): The 2D distance matrix from which the clusters
            are calculated.
        threshold(float): The epsilon threshold value for the clustering
        min_samples(int): The minimum allowed cluster size.

    Returns:
        list: A list of clusters, where each cluster is a list of indices for
        the elements belonging to the cluster.
    """
    # As the distance have been normalized with respect to the covalent radiis,
    # the distance matrix may in some cases have negative values. This simply
    # means that the distance between two atoms is smaller than the sum of
    # their covalent radiis. The distance values are here clipped to
    # zero to avoid problems in the cluster detection. Another option would be
    # to normalize the distances so that unity would correspond to the sum of
    # the two radii.
    np.clip(dist_matrix, a_min=0, a_max=None, out=dist_matrix)

    # Detect clusters
    db = DBSCAN(eps=threshold, min_samples=min_samples, metric='precomputed', n_jobs=1)
    db.fit(dist_matrix)
    clusters = db.labels_

    # Make a list of the different clusters. All item with cluster = -1 will
    # get a separate group.
    cluster_groups = []
    group_map = defaultdict(list)
    for i_atom, i_clust in enumerate(clusters):
        if i_clust == -1:
            cluster_groups.append([i_atom])
        else:
            group_map[i_clust].append(i_atom)
    cluster_groups.extend(group_map.values())

    return cluster_groups


def get_covalent_distances(system, mic=True):
    """Returns a distance matrix where the covalent radii have been taken into
    account. Clips negative values to be zero.
    """
    if system is not None:
        if len(system) == 1:
            return np.array([[0]])

        # Calculate distance matrix with radii taken into account
        distance_matrix = system.get_all_distances(mic=mic)

    # Remove the radii from distances and clip out negative values
    numbers = system.get_atomic_numbers()
    radii = covalent_radii[numbers]
    radii_matrix = radii[None, :] + radii[:, None]
    distance_matrix -= radii_matrix
    np.clip(distance_matrix, 0, None, out=distance_matrix)

    return distance_matrix


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


def get_distance_matrix(pos1, pos2, cell=None, pbc=None, mic=False, use_self_distance=True):
    """Calculates the distance matrix. If wrap_distances=True, calculates
    the matrix using periodic distances

    Args:
        pos1(np.ndarray): Array of cartesian positions
        pos2(np.ndarray): Array of cartesian positions
        cell(): The 3x3 cell of the system. Needed if minimum image convention
            enabled.
        pbc(): The periodic boundary conditions of the system. Needed if
            minimum image convention enabled.
        mic (bool): Whether to apply minimum image convention fo the distances,
            i.e. the distance to the closest periodic image is returned.
        use_self_distance(boolean): Determines how the distance from atom to
            itself is determined. If True, zero is returned, if false, the
            distance to the closest copy is returned.

    Returns:
        np.ndarray: A :math:`N_{atoms} \times N_{atoms}` matrix of distances.
    """
    disp_tensor = get_displacement_tensor(pos1, pos2, cell, pbc, mic, use_self_distance=use_self_distance)
    distance_matrix = np.linalg.norm(disp_tensor, axis=2)

    return distance_matrix


def get_displacement_tensor(
        pos1,
        pos2,
        cell=None,
        pbc=None,
        mic=False,
        max_distance=None,
        return_factors=False,
        return_distances=False,
        use_self_distance=True
    ):
    """Given an array of positions, calculates the 3D displacement tensor
    between the positions.

    The displacement tensor is a matrix where the entry A[i, j, :] is the
    vector pos1[i] - pos2[j], i.e. the vector from pos2 to pos1

    Args:
        pos1(np.ndarray): 2D array of positions
        pos2(np.ndarray): 2D array of positions
        cell(np.ndarray): Cell for taking into account the periodicity
        pbc(boolean or a list of booleans): Periodicity of the axes
        mic(boolean): Whether to return the displacement to the nearest
            periodic copy
        mic_copies(np.ndarray): The maximum number of periodic copies to
            consider in each direction. If not specified, the maximum possible
            number of copies is determined and used.
        use_self_distance(boolean): Determines how the distance from atom to
            itself is determined. If True, zero is returned, if false, the
            distance to the closest copy is returned.

    Returns:
        np.ndarray: 3D displacement tensor
        (optional) np.ndarray: The indices of the periodic copies in which the
            minimum image was found
        (optional) np.ndarray: The lengths of the displacement vectors
    """
    # Make 1D into 2D
    shape1 = pos1.shape
    shape2 = pos2.shape
    if len(shape1) == 1:
        n_cols1 = len(pos1)
        pos1 = np.reshape(pos1, (-1, n_cols1))
    if len(shape2) == 1:
        n_cols2 = len(pos2)
        pos2 = np.reshape(pos2, (-1, n_cols2))

    # Calculate the pairwise distance vectors
    disp_tensor = pos1[:, None, :] - pos2[None, :, :]
    tensor_shape = np.array(disp_tensor.shape)

    # If minimum image convention is used, get the corrected vectors
    if mic:
        flattened_disp = np.reshape(disp_tensor, (-1, 3))
        D, D_len, factors = find_mic(flattened_disp, cell, pbc, max_distance)
        disp_tensor = np.reshape(D, tensor_shape)
        if return_factors:
            factors = np.reshape(factors, tensor_shape)
        if return_distances:
            lengths = np.reshape(D_len, (tensor_shape[0], tensor_shape[1]))
    else:
        if return_factors:
            factors = np.zeros((tensor_shape[0], tensor_shape[1], 3))
        if return_distances:
            lengths = np.linalg.norm(disp_tensor, axis=2)

    if return_factors and return_distances:
        return disp_tensor, factors, lengths
    elif return_factors:
        return disp_tensor, factors
    elif return_distances:
        return disp_tensor, lengths
    else:
        return disp_tensor


def find_mic(D, cell, pbc, max_distance=None):
    """Finds the minimum-image representation of vector(s) D.

    Args:
        D(np.ndarray): A nx3 array of vectors.
        cell(np.ndarray): A valid ase Atoms cell definition.
        pbc(np.ndarray): A 3x1 boolean array for the periodic boundary
            conditions.
        mic_copies(np.ndarray): The maximum number of periodic copies to
            consider in each direction. If not specified, the maximum possible
            number of copies is determined and used.

    Returns:
        np.ndarray: The minimum image versions of the given vectors.
        np.ndarray: The shifts corresponding to the indices of the neighbouring
            cells in which the vectors were found.
    """
    cell = ase.geometry.complete_cell(cell)
    n_vectors = len(D)

    # Calculate the 4 unique unit cell diagonal lengths
    diags = np.sqrt((np.dot([[1, 1, 1],
                             [-1, 1, 1],
                             [1, -1, 1],
                             [-1, -1, 1],
                             ], cell)**2).sum(1))

    Dr = np.dot(D, np.linalg.inv(cell))

    # calculate 'mic' vectors (D) and lengths (D_len) using simple method
    # return mic vectors and lengths for only orthorhombic cells,
    # as the results may be wrong for non-orthorhombic cells
    if (max(diags) - min(diags)) / max(diags) < 1e-9:

        factors = np.floor(Dr + 0.5)*pbc
        D = np.dot(Dr - factors, cell)
        D_len = np.linalg.norm(D, axis=1)
        return D, D_len, factors
    # Non-orthorhombic cell
    else:
        D_len = np.linalg.norm(D, axis=1)

    # The cutoff radius is the longest direct distance between atoms
    # or half the longest lattice diagonal, whichever is smaller
    if max_distance is None:
        max_distance = max(D_len)
    mic_cutoff = min(max_distance, max(diags) / 2.)

    # Construct a list of translation vectors. For example, if we are
    # searching only the nearest images (27 total), tvecs will be a
    # 27x3 array of translation vectors.
    tvec_factors = get_neighbour_cells(cell, mic_cutoff, pbc)
    tvecs = np.dot(tvec_factors, cell)

    # Translate the direct displacement vectors by each translation
    # vector, and calculate the corresponding lengths.
    D_trans = tvecs[np.newaxis] + D[:, np.newaxis]
    D_trans_len = np.linalg.norm(D_trans, axis=2)

    # Find mic distances and corresponding vector(s) for each given pair
    # of atoms. For symmetrical systems, there may be more than one
    # translation vector corresponding to the MIC distance; this finds the
    # first one in D_trans_len.
    D_min_ind = D_trans_len.argmin(axis=1)
    D_min_len = D_trans_len[list(range(n_vectors)), D_min_ind]
    D_min = D_trans[list(range(n_vectors)), D_min_ind]
    factors = tvec_factors[D_min_ind, :]

    # Negate the factors because of the order of the displacement
    factors *= -1

    return D_min, D_min_len, factors


def get_neighbour_cells(cell, cutoff, pbc):
    """Given a cell and a cutoff, returns the indices of the copies of the cell
    which have to be searched in order to reach atom within the cutoff
    distance.

    The number of neighboring images to search in each direction is equal to
    the ceiling of the cutoff distance (defined above) divided by the length of
    the projection of the lattice vector onto its corresponding surface normal.
    a's surface normal vector is e.g.  b x c / (|b| |c|), so this projection is
    (a . (b x c)) / (|b| |c|).  The numerator is just the lattice volume, so
    this can be simplified to V / (|b| |c|). This is rewritten as V |a| / (|a|
    |b| |c|) for vectorization purposes.

    Args:

    Returns:
    """
    pbc = expand_pbc(pbc)
    latt_len = np.linalg.norm(cell, axis=1)
    V = abs(np.linalg.det(cell))
    mic_copies = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                            (V * latt_len)), dtype=int)
    n0 = range(-mic_copies[0], mic_copies[0] + 1)
    n1 = range(-mic_copies[1], mic_copies[1] + 1)
    n2 = range(-mic_copies[2], mic_copies[2] + 1)

    factors = cartesian((n0, n1, n2))

    return factors


def get_mic_vector(w, v, cell):
    """Used to calculate the minimum image version of a vector in the given
    cell.

    Args:
        rel_vector(np.ndarray): Relative vector in the cell basis:

    Returns:
        np.ndarray: The MIC version of the given vector
        np.ndarray: The shift that corresponds to the neighbouring cell in
            which the copy was found.
    """
    # Get the original shift
    dvw = w-v
    cart_vec = to_cartesian(cell, dvw)

    orig_shift = np.array([0, 0, 0])
    for i in range(0, 3):
        if dvw[i] < -0.5 or dvw[i] > 0.5:
            k = np.floor(dvw[i] + 0.5)
            dvw[i] -= k
            orig_shift[i] = -k

    # Figure out in which block of 2x2x2 cells we are testing
    init_val = np.array([0, 0, 0])
    init_val[dvw > 0] = -1

    # Figure out the minimum vector in this block of 8 cells
    a_range = range(init_val[0], init_val[0]+2)
    b_range = range(init_val[1], init_val[1]+2)
    c_range = range(init_val[2], init_val[2]+2)
    multipliers = cartesian((a_range, b_range, c_range))

    vec_copies = cart_vec + np.dot(multipliers, cell)
    squared_len = np.sum(vec_copies**2)
    min_index = np.argmin(squared_len)
    min_shift = multipliers[min_index]

    shift = orig_shift + min_shift

    addition = np.dot(shift, cell)
    result = cart_vec + addition

    return result, shift


def expand_pbc(pbc):
    """Used to expand a pbc definition into an array of three booleans.

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


def get_positions_within_basis(
        system,
        basis,
        origin,
        tolerance_low,
        tolerance_high,
        mask=[True, True, True],
        pbc=True
    ):
    """Used to return the indices of positions that are inside a certain basis.
    Also takes periodic boundaries into account.

    Args:
        system(ASE.Atoms): System from which the positions are searched.
        basis(np.ndarray): New basis vectors.
        origin(np.ndarray): New origin of the basis in cartesian coordinates.
        tolerance(float): The tolerance for the end points of the cell.
        mask(sequence of bool): Mask for selecting the basis's to consider.
        pbc(sequence of bool): The periodicity of the system.

    Returns:
        sequence of int: Indices of the atoms within this cell in the given
            system.
        np.ndarray: Relative positions of the found atoms.
        np.ndarray: The index of the periodic copy in which the position was
            found.
    """
    # If the search extend beyound the cell boundary and periodic boundaries
    # allow, we must divide the search area into multiple regions

    # Transform positions into the new basis
    cart_pos = system.get_positions()
    orig_cell = system.get_cell()
    pbc = expand_pbc(pbc)

    # We need to expand the system in different directions to get the

    # See if the new positions extend beyound the boundaries. The original
    # simulation cell is always convex, so we can just check the corners of
    # unit cell defined by the basis
    max_a = origin + basis[0, :]
    max_b = origin + basis[1, :]
    max_c = origin + basis[1, :]
    max_ab = origin + basis[0, :] + basis[1, :]
    max_ac = origin + basis[0, :] + basis[2, :]
    max_bc = origin + basis[1, :] + basis[2, :]
    max_abc = origin + basis[0, :] + basis[1, :] + basis[2, :]
    vectors = np.array((max_a, max_b, max_c, max_ab, max_ac, max_bc, max_abc))
    rel_vectors = to_scaled(orig_cell, vectors, wrap=False, pbc=system.get_pbc())
    factors = np.floor(rel_vectors).astype(int)
    min_factors = np.min(factors, axis=0)
    max_factors = np.max(factors, axis=0)
    a_range = range(min_factors[0], max_factors[0]+1)
    b_range = range(min_factors[1], max_factors[1]+1)
    c_range = range(min_factors[2], max_factors[2]+1)
    factors = matid.geometry.cartesian((a_range, b_range, c_range))

    directions = []
    for factor in factors:
        a_per = factor[0]
        b_per = factor[1]
        c_per = factor[2]
        allow = True
        if a_per != 0 and not pbc[0]:
            allow = False
        if b_per != 0 and not pbc[1]:
            allow = False
        if c_per != 0 and not pbc[2]:
            allow = False
        if allow:
            directions.append(factor)

    # If the new cell is overflowing beyound the boundaries of the original
    # system, we have to also check the periodic copies.
    indices = []
    a_prec_low, b_prec_low, c_prec_low = tolerance_low/np.linalg.norm(basis, axis=1)
    a_prec_high, b_prec_high, c_prec_high = tolerance_high/np.linalg.norm(basis, axis=1)
    orig_basis = system.get_cell()
    cell_pos = []
    factors = []
    for i_dir in directions:

        vec_new_cart = cart_pos + np.dot(i_dir, orig_basis)
        vec_new_rel = change_basis(vec_new_cart - origin, basis)

        # If no positions are defined, find the atoms within the cell
        for i_pos, pos in enumerate(vec_new_rel):
            if mask[0]:
                x = 0 - a_prec_low <= pos[0] <= 1 - a_prec_high
            else:
                x = True
            if mask[1]:
                y = 0 - b_prec_low <= pos[1] <= 1 - b_prec_high
            else:
                y = True
            if mask[2]:
                z = 0 - c_prec_low <= pos[2] <= 1 - c_prec_high
            else:
                z = True

            if x and y and z:
                indices.append(i_pos)
                cell_pos.append(pos)
                factors.append(i_dir)

    cell_pos = np.array(cell_pos)
    indices = np.array(indices)
    factors = np.array(factors)

    return indices, cell_pos, factors


def get_matches(
        system,
        positions,
        numbers,
        tolerances,
        mic=True
    ):
    """Given a system and a list of cartesian positions and atomic numbers,
    returns a list of indices for the atoms corresponding to the given
    positions with some tolerance.

    Args:
        system(ASE.Atoms): System where to search the positions
        positions(np.ndarray): Positions to match in the system.
        tolerances(np.ndarray): Maximum allowed distance for each vector that
            is allowed for a match in position.
        mic(boolean): Whether to find the minimum image copy.

    Returns:
        np.ndarray: indices of matched atoms
        list: list of substitutions
        list: list of vacancies
        np.ndarray: for each searched position, an integer array representing
            the number of the periodic copy where the match was found.
    """
    orig_num = system.get_atomic_numbers()
    orig_pos = system.get_positions()
    cell = system.get_cell()
    pbc = system.get_pbc()
    pbc = expand_pbc(pbc)
    scaled_pos2 = to_scaled(cell, positions, wrap=False)

    disp_tensor, factors, dist_matrix = get_displacement_tensor(
        positions,
        orig_pos,
        cell,
        pbc,
        mic=mic,
        max_distance=tolerances.max(),
        return_factors=True,
        return_distances=True
    )

    # Find the closest atom within the tolerance and with the required atomic
    # number, or if not found, get the closest atom that is within the
    # tolerance
    best_matches = []
    best_substitutions = []
    for i_atom, i in enumerate(dist_matrix):
        near_mask = i <= tolerances[i_atom]
        element_mask = orig_num == numbers[i_atom]
        combined_mask = near_mask & element_mask
        possible_indices = np.where(combined_mask)[0]
        if len(possible_indices) != 0:
            min_dist_index = np.argmin(i[combined_mask])
            best_index = possible_indices[min_dist_index]
            best_matches.append(best_index)
            best_substitutions.append(None)
        elif near_mask.any():
            best_matches.append(None)
            near_indices = np.where(near_mask)[0]
            nearest_index = np.argmin(i[near_mask])
            best_substitutions.append(near_indices[nearest_index])
        else:
            best_matches.append(None)
            best_substitutions.append(None)

    # min_ind = np.argmin(dist_matrix, axis=1)
    matches = []
    substitutions = []
    vacancies = []
    copy_indices = []

    for i, (i_match, i_subst) in enumerate(zip(best_matches, best_substitutions)):

        match = None
        copy = None
        subst = None
        b_num = numbers[i]

        if i_match is not None:
            ind = i_match
            match = ind

            # If a match was found the factor is reported based on the
            # displacement tensor
            i_move = factors[i][ind]
            copy = i_move
        elif i_subst is not None:
            ind = i_subst
            a_num = orig_num[ind]

            # Wrap the substitute position
            subst_pos_cart = orig_pos[ind]
            subst = Substitution(ind, subst_pos_cart, b_num, a_num)

            # If a match was found the factor is reported based on the
            # displacement tensor
            i_move = factors[i][ind]
            copy = i_move
        else:
            vacancies.append(Atom(b_num, position=positions[i]))

            # If no match was found, the factor is reported from the scaled
            # positions
            copy = np.floor(scaled_pos2[i])

        substitutions.append(subst)
        matches.append(match)
        copy_indices.append(copy)

    return matches, substitutions, vacancies, copy_indices


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
    # Force 1D to 2D and check shape
    shape = positions.shape
    if len(shape) == 1:
        positions = positions[None, :]
        shape = positions.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(
            "The given positions are not compatible. Please provide positions "
            "as rows of a two-dimensional array."
        )
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
    """Used to transform a set of relative positions to the cartesian basis
    defined by the cell of this system.

    Args:
        cell (numpy.ndarray): 3x3 array with lattice vectors as rows.
        positions (numpy.ndarray): The positions to scale. These postiions
            should have a shape of [n, 3], where n is the number of positions.
        wrap (numpy.ndarray): Whether the positions should be wrapped
            inside the cell.

    Returns:
        numpy.ndarray: The cartesian positions
    """
    # Force 1D to 2D and check shape
    shape = scaled_positions.shape
    if len(shape) == 1:
        scaled_positions = scaled_positions[None, :]
        shape = scaled_positions.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(
            "The given positions are not compatible. Please provide positions "
            "as rows of a two-dimensional array."
        )

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


def get_thickness(system, axis):
    """Get the thickness of an atomic system along a basis vector direction.

    Args:
        system(ase.Atoms): The system from which the thickness is evaluated.
        axis(int): The index of the unit cell basis in which direction the
            thickness is evaluated.

    Returns:
        float: The thickness of the system as measured from the center of the
        topmost atom to the center of lowermost atom.
    """
    pos = system.get_scaled_positions()[:, axis]
    pos_min = pos.min()
    pos_max = pos.max()
    basis = system.get_cell()[axis, :]
    thickness = (pos_max - pos_min)*np.linalg.norm(basis)

    return thickness


def get_minimized_cell(system, axis, min_size):
    """Used to resize the cell in the given cell direction so that all atoms
    are within the cell.

    Args:
        system(ase.Atoms): The system to minimize. Must have a cell with
            non-zero volume.
        axis(int): Index of the axis to minimize.
        axis(min_size): The minimum size for the cell in the given direction.

    Returns:
        ase.Atoms: The new minimized system.
    """
    # Grow the cell to fit all atoms
    rel_pos = system.get_scaled_positions(wrap=False)
    num = system.get_atomic_numbers()
    pbc = system.get_pbc()
    basis = system.get_cell()
    c = basis[axis, :]
    c_length = np.linalg.norm(c)
    c_norm = c/c_length
    c_comp = rel_pos[:, axis]
    # print(c_comp)

    min_index = np.argmin(c_comp, axis=0)
    max_index = np.argmax(c_comp, axis=0)

    pos_min_rel = np.zeros((3))
    pos_min_rel[axis] = c_comp[min_index]
    pos_max_rel = np.zeros((3))
    pos_max_rel[axis] = c_comp[max_index]

    pos_min_cart = matid.geometry.to_cartesian(basis, pos_min_rel)
    pos_max_cart = matid.geometry.to_cartesian(basis, pos_max_rel)
    c_real_cart = pos_max_cart-pos_min_cart
    c_size = np.linalg.norm(c_real_cart)

    # We demand a minimum size for the c-vector even if the system seems to
    # be purely 2-dimensional. This is done because the 3D-space cannot be
    # searched properly if one dimension is flat.
    if c_size < min_size:
        c_inflated_cart = min_size*c_norm
        c_new_cart = c_inflated_cart
    else:
        c_new_cart = c_real_cart
    new_basis = np.array(basis)
    new_basis[axis, :] = c_new_cart

    new_scaled_pos = rel_pos - pos_min_rel
    new_cart_test = matid.geometry.to_cartesian(basis, new_scaled_pos)
    new_scaled_pos = matid.geometry.to_scaled(new_basis, new_cart_test)

    # Translate the system to be in the middle if size is smaller than the
    # given minimum size
    if c_size < min_size:
        offset_cart = (c_real_cart-c_inflated_cart)/2
        offset_rel = matid.geometry.to_scaled(new_basis, offset_cart)
        new_scaled_pos -= offset_rel

    # Create translated system
    minimized_system = Atoms(
        cell=new_basis,
        scaled_positions=new_scaled_pos,
        symbols=num,
        pbc=pbc
    )

    return minimized_system


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Args:
        arrays(sequence of arrays): The arrays from which the product is
            created.
        out(ndarray): Array to place the cartesian product in.

    Returns:
        ndarray: 2-D array of shape (M, len(arrays)) containing cartesian
        products formed of input arrays.

    Example:
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def get_crystallinity(symmetry_analyser):
    """Quantifies the crystallinity of the structure as a ratio of symmetries
    per number of unique atoms in primitive cell. This metric can be used to
    distinguish between amorphous and 'regular' crystals.

    The number of symmetry operations corresponds to the symmetry operations
    corresponding to the hall number of the structure. The symmetry operations
    as given by spglib.get_symmetry() are specific to the original structure,
    and they have not been reduced to the symmetries of the space group.

    Args:
        symmetry_analyser (): A SymmetryAnalyzer object that has been
            initialized with an atomic structure.

    Returns:
        (float) A ratio of symmetries per unique atoms in the primitive cell.
    """
    # Get the number of equivalent atoms in the primitive cell.
    n_unique_atoms_prim = len(symmetry_analyser.get_equivalent_atoms_primitive())

    hall_number = symmetry_analyser.get_hall_number()
    sym_ops = spglib.get_symmetry_from_database(hall_number)
    n_symmetries = len(sym_ops["rotations"])

    ratio = n_symmetries/float(n_unique_atoms_prim)

    return ratio


def get_distances(system: Atoms) -> Distances:
    """Returns complete distance information.

    Args:
        system: The system from which distances are calculated from.
    Returns:
        A Distances instance.
    """
    pos = system.get_positions()
    cell = system.get_cell()
    pbc = system.get_pbc()
    disp_tensor_finite = get_displacement_tensor(pos, pos)
    if pbc.any():
        disp_tensor_mic, disp_factors = get_displacement_tensor(
            pos,
            pos,
            cell,
            pbc,
            mic=True,
            return_factors=True
        )
    else:
        disp_tensor_mic = disp_tensor_finite
        disp_factors = np.zeros(disp_tensor_finite.shape)
    dist_matrix_mic = np.linalg.norm(disp_tensor_mic, axis=2)

    # Calculate the distance matrix where the periodicity and the covalent
    # radii have been taken into account
    dist_matrix_radii_mic = np.array(dist_matrix_mic)
    num = system.get_atomic_numbers()
    radii = covalent_radii[num]
    radii_matrix = radii[:, None] + radii[None, :]
    dist_matrix_radii_mic -= radii_matrix

    return Distances(disp_tensor_mic, disp_factors, disp_tensor_finite, dist_matrix_mic, dist_matrix_radii_mic)


def swap_basis(atoms: Atoms, a: int, b: int):
    """Used to swap two bases in a system. The geometry remains identical.

    Args:
        a: First index to swap
        b: Second index to swap
    """
    cell_old = atoms.get_cell()
    pbc_old = atoms.get_pbc()
    cell_new = np.array(cell_old)
    cell_new[a] = cell_old[b]
    cell_new[b] = cell_old[a]
    pbc_new = np.array(pbc_old)
    pbc_new[a] = pbc_old[b]
    pbc_new[b] = pbc_old[a]
    atoms.set_cell(cell_new)
    atoms.set_pbc(pbc_new)


# def get_surface_normal_direction(system):
    # """Used to estimate a normal vector for a 2D like structure.

    # Args:
        # system (ase.Atoms): The system to examine.

    # Returns:
        # np.ndarray: The estimated surface normal vector
    # """
    # repeated = get_extended_system(system, 15)
    # # vectors = system.get_cell()

    # # Get the eigenvalues and eigenvectors of the moment of inertia tensor
    # val, vec = get_moments_of_inertia(repeated)
    # sorted_indices = np.argsort(val)
    # val = val[sorted_indices]
    # vec = vec[sorted_indices]

    # # If the moment of inertia is not significantly bigger in one
    # # direction, then the system cannot be described as a surface.
    # moment_limit = 1.5
    # if val[-1] < moment_limit*val[0] and val[-1] < moment_limit*val[1]:
        # raise ValueError(
            # "The given system could not be identified as a surface. Make"
            # " sure that you provide a surface system with a sufficient"
            # " vacuum gap between the layers (at least ~8 angstroms of vacuum"
            # " between layers.)"
        # )

    # # The biggest component is the orhogonal one
    # orthogonal_dir = vec[-1]

    # return orthogonal_dir

    # # Find out the cell direction that corresponds to the orthogonal one
    # # cell = repeated.get_cell()
    # # dots = np.abs(np.dot(orthogonal_dir, vectors.T))
    # # orthogonal_vector_index = np.argmax(dots)
    # # orthogonal_vector = vectors[orthogonal_vector_index]
    # # orthogonal_dir = orthogonal_vector/np.linalg.norm(orthogonal_vector)

    # return orthogonal_dir
