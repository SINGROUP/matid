import itertools

import numpy as np

from ase.data import covalent_radii
from ase import Atoms
from ase.visualize import view

from sklearn.cluster import DBSCAN

import spglib

from systax.exceptions import ClassificationError
from systax.components import SurfaceComponent, AtomComponent, MoleculeComponent
from systax.classifications import Surface, Atom, Molecule, Crystal
from systax.data.element_data import get_covalent_radii
from systax.core.system import System


class SystemCache(dict):
    pass


class Classifier(object):
    """A class that is used to analyze the contents of an atomistic system and
    separate the consituent atoms into different components along with some
    meaningful additional information.
    """
    def __init__(
            self,
            max_dist=3,
            pos_tol=0.5,
            angle_tol=15,
            n_seeds=2,
            n_repetitions=2):
        self.max_dist = max_dist
        self.pos_tol = pos_tol
        self.angle_tol = angle_tol
        self.n_seeds = 1
        self.n_repetitions = n_repetitions
        self._repeated_system = None
        self._analyzed = False
        self._orthogonal_dir = None

        # self.surfaces = []
        # self.crystals = []
        # self.atoms = []
        # self.molecules = []
        self.decisions = {}

    def classify(self, system):
        """A function that analyzes the system and breaks it into different
        components.

        The component detection works like this:

            - Calculate the space filling to separate bulk systems.
            - Calculate the moments of inertia to separate isotropic systems
              from anisotropic
            - Find out the directions with a vacuum gap
        """
        if isinstance(system, Atoms):
            system = System.from_atoms(system)
        self.system = system
        surface_comps = []
        atom_comps = []
        molecule_comps = []
        crystal_comps = []

        # Run a high level analysis to determine the system type
        periodicity = system.get_pbc()
        n_periodic = np.sum(periodicity)
        # view(extended_system)

        if n_periodic == 0:

            # Check if system has only one atom
            n_atoms = len(system)
            if n_atoms == 1:
                atom_comps.append(AtomComponent([0], system))
            # Check if the the system has one or multiple components
            else:
                clusters = self.get_clusters(system)
                for cluster in clusters:
                    n_atoms_cluster = len(cluster)
                    if n_atoms_cluster == 1:
                        atom_comps.append(AtomComponent(cluster, system[cluster]))
                    elif n_atoms_cluster > 1:
                        molecule_comps.append(MoleculeComponent(cluster, system[cluster]))

        # If the system has at least one periodic dimension, check the periodic
        # directions for a vacuum gap.
        else:
            vacuum_dir = self.find_vacuum_directions(system)
            n_vacuum = np.sum(vacuum_dir)

            # If all directions have a vacuum seaprating the copies, the system
            # represents a finite structure.
            if n_vacuum == 3:

                # Check if system has only one atom
                n_atoms = len(system)
                if n_atoms == 1:
                    atom_comps.append(AtomComponent([0], system))

                # Check if the the system has one or multiple components.
                else:
                    clusters = self.get_clusters(system)
                    for cluster in clusters:
                        n_atoms_cluster = len(cluster)
                        if n_atoms_cluster == 1:
                            atom_comps.append(AtomComponent(cluster, system[cluster]))
                        elif n_atoms_cluster > 1:
                            molecule_comps.append(MoleculeComponent(cluster, system[cluster]))

            # Bulk structures
            # if n_vacuum == 0:

        # elif n_periodic == 2:

        # elif n_periodic == 3:

            # # Find directions with a vacuum gap
            # vacuum_dir = self.find_vacuum_directions(system)
            # n_vacuum = np.sum(vacuum_dir)

            # if n_vacuum == 3:

                # # Check if system has only one atom
                # n_atoms = len(system)
                # if n_atoms == 1:
                    # atom_comps.append(AtomComponent([0], system))

                # # Check if the the system has one or multiple components.
                # else:
                    # clusters = self.get_clusters(system)
                    # for cluster in clusters:
                        # n_atoms_cluster = len(cluster)
                        # if n_atoms_cluster == 1:
                            # atom_comps.append(AtomComponent(cluster, system[cluster]))
                        # elif n_atoms_cluster > 1:
                            # molecule_comps.append(MoleculeComponent(cluster, system[cluster]))

        # Return a classification for this system.
        n_molecules = len(molecule_comps)
        n_atoms = len(atom_comps)
        n_crystals = len(crystal_comps)
        n_surfaces = len(surface_comps)
        if (n_atoms == 1) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_surfaces == 0):

            return Atom(atoms=atom_comps)

        elif (n_atoms == 0) and \
                (n_molecules == 1) and \
                (n_crystals == 0) and \
                (n_surfaces == 0):
            return Molecule(molecules=molecule_comps)

        elif (n_atoms == 0) and \
                (n_molecules == 0) and \
                (n_crystals == 1) and \
                (n_surfaces == 0):
            return Crystal(crystals=crystal_comps)

            # occupied_ratio = self.get_space_filling(system)
            # if occupied_ratio >= 0.7:
                # self.crystals.append(Crystal())

            # extended_system = self.get_extended_system(system, 15)
            # moments, axes = self.get_inertia_tensor(extended_system, weight=False)
            # n_clusters = len(clusters)

        # Find the different clusters
        # cluster_indices = self.get_clusters(system)

        # # Run the surface detection on each cluster (we do not initially know
        # # which one is the surface.
        # # system = self._get_repeated_system()
        # misc_indices = []
        # for indices in cluster_indices:
            # cluster_system = self.system[indices]
            # i_surf = self._find_surfaces(cluster_system, indices)

            # surface_indices = []
            # for surface in i_surf:
                # i_surf_ind = surface.indices
                # surface_indices.extend(i_surf_ind)

            # indices = set(indices)
            # surface_indices = set(surface_indices)
            # i_misc_ind = indices - surface_indices
            # if i_misc_ind:
                # misc_indices.append(list(i_misc_ind))

            # surfaces.extend(i_surf)

        # # Create a combined system for all the atoms that were not categorized
        # # yet.
        # if len(misc_indices) != 0:
            # misc_system = Atoms()
            # misc_orig_indices = []
            # misc_system.set_cell(system.get_cell())
            # for misc_ind in misc_indices:
                # i_misc_sys = system[misc_ind]
                # misc_orig_indices.extend(misc_ind)
                # misc_system += i_misc_sys

            # # Find clusters
            # misc_orig_indices = np.array(misc_orig_indices)
            # cluster_indices = self.get_clusters(misc_system)

            # # Categorize clusters as molecules or atoms
            # # atoms = []
            # # molecules = []
            # for cluster in cluster_indices:
                # n_atoms_clust = len(cluster)
                # orig_indices = misc_orig_indices[cluster]
                # if n_atoms_clust == 1:
                    # atom = Atom(orig_indices, system[orig_indices])
                    # self.atoms.append(atom)
                # elif n_atoms_clust > 1:
                    # molecule = Molecule(orig_indices, system[orig_indices])
                    # self.molecules.append(molecule)

    def handle_pbc_0(self, system):
        """
        """

    def handle_pbc_1(self, system):
        """
        """

    def handle_pbc_2(self, system):
        """
        """

    def handle_pbc_3(self, system):
        """
        """

    def get_extended_system(self, system, target_size):
        """Replicate the system in different directions to reach a suitable
        system size for getting the moments of inertia.

        Args:
            system (ase.Atoms): The original system.
            target_size (float): The target size for the extended system.

        Returns:
            ase.Atoms: The extended system.
        """
        cell = system.get_cell()
        sizes = np.linalg.norm(cell, axis=1)
        repetitions = np.maximum(np.round(sizes/target_size), 1).astype(int)
        extended_system = system.repeat(repetitions)

        return extended_system

    def get_space_filling(self, system):
        """Calculates the ratio of vacuum to filled space by assuming covalent
        radii for the atoms.

        Args:
            system(ASE Atoms): Atomic system.

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

    def get_inertia_tensor(self, system, weight=True):
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
        centroid = self.get_center_of_mass(system, weight)
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

    def find_vacuum_directions(self, system, threshold=7.0):
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

    def get_center_of_mass(self, system, weight=True):
        """
        """
        positions = system.get_positions()
        if weight:
            weights = system.get_masses()
        else:
            weights = np.ones((len(system)))
        cm = np.dot(weights, positions/weights.sum())

        return cm

    def get_clusters(self, system):
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

    def _find_surfaces(self, system, orig_indices):
        """
        """
        # Find the seed points
        seed_points = self._find_seed_points(system)

        # Create a cache for reused quantities
        syscache = SystemCache()
        syscache["system"] = system
        syscache["positions"] = system.get_positions()

        # Calculate displacement tensor
        pos = np.array(system.get_positions())
        disp_tensor = pos[:, None, :] - pos[None, :, :]
        syscache["disp_tensor"] = disp_tensor

        # Find possible bases for each seed point
        cell_basis_vectors = None
        for seed_index in seed_points:
            possible_spans = self._find_possible_spans(syscache, seed_index)
            if len(possible_spans) >= 3:
                valid_spans = self._find_valid_spans(syscache, seed_index, possible_spans)
                if len(valid_spans) >= 3:
                    lin_ind_spans = self._find_optimal_span(syscache, valid_spans)
                    if len(lin_ind_spans) == 3:
                        cell_basis_vectors = lin_ind_spans
                        break

        if cell_basis_vectors is None:
            return []

        # Find the atoms within the found cell
        cell_pos, cell_numbers = self._find_cell_atoms(system, seed_index, cell_basis_vectors)

        # print(cell_basis_vectors)
        # print(cell_pos)
        # print(cell_numbers)

        # Find the standardized cell
        material = (cell_basis_vectors, cell_pos, cell_numbers)
        # The symprec in spglib is a diameter, thus the factor 2
        symmetry_dataset = spglib.get_symmetry_dataset(material, symprec=2*self.pos_tol)
        if symmetry_dataset is None:
            raise ClassificationError("Error in finding the symmetry dataset of a cell with SPGLIB.")

        # Get the normalized system
        # print(cell_basis_vectors)
        # print(cell_pos)
        normalizer = CellNormalizer()
        bulk_system = normalizer.normalize(
            symmetry_dataset,
            cell_pos,
            cell_basis_vectors,
            2*self.pos_tol)

        # print(symmetry_dataset["std_positions"])
        # print(symmetry_dataset["std_lattice"])

        # Find the adsorbate by looking at translational symmetry. Everything
        # else that does not belong to the surface unit cell is considered an
        # adsorbate.
        surface_indices = self._find_adsorbent(system, seed_index, cell_pos, cell_numbers, cell_basis_vectors)

        # Find the original indices for the surface
        surface_atoms = system[surface_indices]
        orig_surface_indices = []
        for index in list(surface_indices):
            orig_surface_indices.append(orig_indices[index])

        surface = SurfaceComponent(
            orig_surface_indices,
            surface_atoms,
            bulk_system,
            symmetry_dataset
        )

        return [surface]

    def _find_surface_recursively(
            self,
            number_to_index_map,
            number_to_pos_map,
            indices,
            cells,
            i,
            j,
            k,
            system,
            seed_index,
            seed_pos,
            seed_number,
            cell_basis,
            cell_pos,
            cell_numbers):
        """A recursive function for traversing the surface and gathering
        indices of the surface atoms.
        """
        # Check if this cell has already been searched
        if (i, j, k) in cells:
            return
        else:
            cells[(i, j, k)] = True

        # Transform positions to the cell basis
        positions = system.get_positions()
        pos_shifted = positions - seed_pos
        basis_inverse = np.linalg.inv(cell_basis.T)
        vec_new = np.dot(pos_shifted, basis_inverse.T)

        # For each atom in the basis, find corresponding atom if possible
        new_indices = []
        for i_pos, pos in enumerate(cell_pos):
            # i_number = cell_numbers[i_pos]
            # possible_pos = number_to_pos_map[i_number]
            disp_tensor = vec_new - pos[np.newaxis, :]
            disp_tensor_cartesian = np.dot(disp_tensor, cell_basis.T)
            dist = np.linalg.norm(disp_tensor_cartesian, axis=1)
            # The tolerance is double here to take into account the possibility
            # that two neighboring cells might be offset from the original cell
            # in opposite directions
            index, = np.where(dist <= self.pos_tol)
            if len(index) != 0:
                new_indices.append(index[0])

        # Add the newly found indices
        if len(new_indices) != 0:
            indices.extend(new_indices)

        # If the seed atom was not found for this cell, end the search
        if seed_index is None:
            return

        # Get the vectors that span from the seed to all other atoms
        # atomic_numbers = system.get_atomic_numbers()
        # atomic_numbers = np.delete(atomic_numbers, (seed_index), axis=0)
        # seed_element = atomic_numbers[seed_index]
        # identical_elem_mask = (atomic_numbers == seed_element)
        # seed_span_lengths = np.linalg.norm(seed_spans, axis=1)
        # distance_mask = (seed_span_lengths < 2*self.max_dist)
        # combined_mask = (distance_mask) & (identical_elem_mask)
        # spans = seed_spans[combined_mask]

        # Find the new seed indices and positions
        new_seed_indices = []
        new_seed_pos = []
        multipliers = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        possible_seed_pos = number_to_pos_map[seed_number]
        possible_seed_indices = number_to_index_map[seed_number]
        for multiplier in multipliers:
            disloc = np.sum(multiplier.T[:, np.newaxis]*cell_basis, axis=0)
            seed_guess = seed_pos + disloc
            i_seed_disp = possible_seed_pos - seed_guess
            i_seed_dist = np.linalg.norm(i_seed_disp, axis=1)
            index = np.where(i_seed_dist <= 2*self.pos_tol)
            index = index[0]
            new_seed_indices.append(index)
            if len(index) != 0:
                i_seed_pos = possible_seed_pos[index[0]]
            else:
                i_seed_pos = seed_guess
            new_seed_pos.append(i_seed_pos)

        # Update the cell for this seed point. This adds more noise to the cell
        # basis vectors but allows us to track curved lattices. The indices 22,
        # 16, and 14 are the indices for the [1,0,0], [0,1,0] and [0, 0, 1]
        # multipliers.
        i_cell = np.vstack((new_seed_pos[22], new_seed_pos[16], new_seed_pos[14]))
        i_cell = i_cell - seed_pos

        # Use the newly found indices to track down new indices with an updated
        # cell.
        for i_seed, multiplier in enumerate(multipliers):
            i_seed_pos = new_seed_pos[i_seed]
            i_seed_index = new_seed_indices[i_seed]

            n_indices = len(i_seed_index)
            # if n_indices > 1:
                # raise ValueError("Too many options when searching for an atom.")
            if n_indices == 0:
                i_seed_index = None
            else:
                i_seed_index = possible_seed_indices[i_seed_index[0]]

            a = i + multiplier[0]
            b = j + multiplier[1]
            c = k + multiplier[2]

            self._find_surface_recursively(number_to_index_map, number_to_pos_map, indices, cells, a, b, c, system, i_seed_index, i_seed_pos, seed_number, i_cell, cell_pos, cell_numbers)

    def _find_adsorbent(self, system, seed_index, cell_pos, cell_numbers, cell_basis):
        """
        """
        # from ase.visualize import view
        # view(system)

        positions = system.get_positions()
        atomic_numbers = system.get_atomic_numbers()
        seed_pos = positions[seed_index][np.newaxis, :]
        seed_number = atomic_numbers[seed_index]
        indices = []
        cells = {}

        # Create a map between an atomic number and indices in the system
        number_to_index_map = {}
        number_to_pos_map = {}
        atomic_number_set = set(atomic_numbers)
        for number in atomic_number_set:
            number_indices = np.where(atomic_numbers == number)[0]
            number_to_index_map[number] = number_indices
            number_to_pos_map[number] = positions[number_indices]

        self._find_surface_recursively(number_to_index_map, number_to_pos_map, indices, cells, 0, 0, 0, system, seed_index, seed_pos, seed_number, cell_basis, cell_pos, cell_numbers)

        return indices

    def _find_possible_spans(self, syscache, seed_index):
        """Finds all the possible vectors that might span a cell.
        """
        # Get the vectors that span from the seed to all other atoms
        disp_tensor = syscache["disp_tensor"]
        system = syscache["system"]

        seed_spans = disp_tensor[:, seed_index]
        atomic_numbers = system.get_atomic_numbers()

        # Remove notion to to self
        seed_spans = np.delete(seed_spans, (seed_index), axis=0)
        atomic_numbers = np.delete(atomic_numbers, (seed_index), axis=0)

        # Find indices of atoms that are identical to seed atom
        seed_element = atomic_numbers[seed_index]
        identical_elem_mask = (atomic_numbers == seed_element)

        # Only keep spans that are smaller than the maximum vector length
        seed_span_lengths = np.linalg.norm(seed_spans, axis=1)
        distance_mask = (seed_span_lengths < self.max_dist)

        # Form a combined mask and filter spans with it
        combined_mask = (distance_mask) & (identical_elem_mask)
        spans = seed_spans[combined_mask]

        return spans

    def _find_valid_spans(self, syscache, seed_index, possible_spans):
        """Check which spans in the given list actually are translational bases
        on the surface.

        In order to be a valid span, there has to be at least one repetition of
        this span for all atoms that are nearby the seed atom.
        """
        # Find atoms that are nearby the seed atom.
        system = syscache["system"]
        disp_tensor = syscache["disp_tensor"]
        positions = syscache["positions"]

        # from ase.visualize import view
        # view(system)

        span_lengths = np.linalg.norm(possible_spans, axis=1)
        max_span_len = np.max(span_lengths)
        seed_pos = positions[seed_index]
        seed_spans = disp_tensor[:, seed_index]
        seed_dist = np.linalg.norm(seed_spans, axis=1)
        neighbour_indices, = np.where(seed_dist < max_span_len)
        neighbour_pos = positions[neighbour_indices]

        # Ensure that only neighbors that are within the "region" defined by
        # the possible spans are included in the test. Neighbor atoms outside
        # the possible spans might already include e.g. adsorbate atoms.
        spans_norm = possible_spans/span_lengths[:, np.newaxis]
        span_dots = np.inner(spans_norm, spans_norm)
        combos = []

        # For each span find the two other spans that are closests, and not
        # already taken
        n_spans = len(possible_spans)
        for i_span in range(n_spans):
            dots = span_dots[i_span, :]
            indices = np.argsort(dots)

            # Combinations are emitted in sorted order
            combinations = itertools.combinations(range(-1, -n_spans, -1), 2)
            for i, j in combinations:
                closest_1 = indices[i]
                closest_2 = indices[j]
                if closest_1 == i_span or closest_2 == i_span:
                    continue
                i_combo = set((i_span, closest_1, closest_2))
                if i_combo not in combos:
                    combos.append(list(i_combo))
                    break

        # Find the neighbors that are within the cells defined by the span combinations
        true_neighbor_indices = []
        shifted_neighbor_pos = neighbour_pos-seed_pos
        for i_combo, combo in enumerate(combos):
            cell = possible_spans[np.array(combo)]
            try:
                inv_cell = np.linalg.inv(cell.T)
            except np.linalg.linalg.LinAlgError:
                continue
            neigh_pos_combo = np.dot(shifted_neighbor_pos, inv_cell.T)

            for i_pos, pos in enumerate(neigh_pos_combo):
                x = 0 <= pos[0] <= 1
                y = 0 <= pos[1] <= 1
                z = 0 <= pos[2] <= 1
                if x and y and z:
                    true_neighbor_indices.append(neighbour_indices[i_pos])

        true_neighbor_indices = set(true_neighbor_indices)
        true_neighbor_indices.discard(seed_index)
        neighbour_indices = list(true_neighbor_indices)
        neighbour_pos = positions[neighbour_indices]

        # Calculate the positions that come from adding or subtracting a
        # possible span
        added = neighbour_pos[:, np.newaxis, :] + possible_spans[np.newaxis, :, :]
        subtr = neighbour_pos[:, np.newaxis, :] - possible_spans[np.newaxis, :, :]

        # Check if a matching atom was found in either the added or subtracted
        # case with some tolerance. We need to take into account the periodic
        # boundary conditions when comparing distances. This is done by
        # checking if there is a closer mirror image.
        added_displ = added[:, :, np.newaxis, :] - positions[np.newaxis, np.newaxis, :, :]
        subtr_displ = subtr[:, :, np.newaxis, :] - positions[np.newaxis, np.newaxis, :, :]

        # Take periodicity into account by wrapping coordinate elements that are
        # bigger than 0.5 or smaller than -0.5
        cell = system.get_cell()
        inverse_cell = np.linalg.inv(cell)

        rel_added_displ = np.dot(added_displ, inverse_cell.T)
        indices = np.where(rel_added_displ > 0.5)
        rel_added_displ[indices] = 1 - rel_added_displ[indices]
        indices = np.where(rel_added_displ < -0.5)
        rel_added_displ[indices] = rel_added_displ[indices] + 1
        added_displ = np.dot(rel_added_displ, cell.T)

        rel_subtr_displ = np.dot(subtr_displ, inverse_cell.T)
        indices = np.where(rel_subtr_displ > 0.5)
        rel_subtr_displ[indices] = 1 - rel_subtr_displ[indices]
        indices = np.where(rel_subtr_displ < -0.5)
        rel_subtr_displ[indices] = rel_subtr_displ[indices] + 1
        subtr_displ = np.dot(rel_subtr_displ, cell.T)

        added_dist = np.linalg.norm(added_displ, axis=3)
        subtr_dist = np.linalg.norm(subtr_displ, axis=3)

        # For every neighbor, and every span, there should be one atom that
        # matches either the added or subtracted span if the span is to be
        # valid
        a_neigh_ind, a_span_ind, _ = np.where(added_dist < 2*self.pos_tol)
        s_neigh_ind, s_span_ind, _ = np.where(subtr_dist < 2*self.pos_tol)
        neighbor_valid_ind = np.concatenate((a_neigh_ind, s_neigh_ind))
        span_valid_ind = np.concatenate((a_span_ind, s_span_ind))

        # Go through the spans and see which ones have a match for every
        # neighbor
        valid_spans = []
        valid_span_indices = []
        neighbor_index_set = set(range(len(neighbour_indices)))
        for span_index in range(len(possible_spans)):
            indices = np.where(span_valid_ind == span_index)
            i_neighbor_ind = neighbor_valid_ind[indices]
            i_neighbor_ind_set = set(i_neighbor_ind.tolist())
            if i_neighbor_ind_set == neighbor_index_set:
                valid_span_indices.append(span_index)

        valid_spans = possible_spans[valid_span_indices]
        valid_spans_length = span_lengths[valid_span_indices]
        valid_spans_dot = span_dots[valid_span_indices]
        syscache["valid_spans"] = valid_spans
        syscache["valid_spans_length"] = valid_spans_length
        syscache["valid_spans_dot"] = valid_spans_dot

        return valid_spans

    def _find_optimal_span(self, syscache, spans):
        """There might be more than three valid spans that were found, so this
        function is used to select three.

        The selection is based on the minimization of the following term:

            \min e_i^2 + \sum_{i \neq j} \hat{e}_i \cdot \hat{e}_j

        where the vectors e are the possible unit vectors
        """
        # print(spans)

        # Get all triplets of spans (combinations)
        span_indices = range(len(spans))
        indices = np.array(list(itertools.combinations(span_indices, 3)))

        # Calculate the 3D array of combined span weights for every triplet
        norms = syscache["valid_spans_length"]
        norm1 = norms[indices[:, 0]]
        norm2 = norms[indices[:, 1]]
        norm3 = norms[indices[:, 2]]
        norm_vector = norm1 + norm2 + norm3
        # print(norm_vector.shape)

        # Calculate the orthogonality tensor from the dot products
        dots = syscache["valid_spans_dot"]
        dot1 = np.abs(dots[indices[:, -1], indices[:, 1]])
        dot2 = np.abs(dots[indices[:, 1], indices[:, 2]])
        dot3 = np.abs(dots[indices[:, 0], indices[:, 2]])
        ortho_vector = dot1 + dot2 + dot3
        # print(ortho_vector.shape)

        # Create a combination of the norm tensor and the dots tensor with a
        # possible weighting
        norm_weight = 1
        ortho_weight = 1
        sum_vector = norm_weight*norm_vector + ortho_weight*ortho_vector

        # Sort the triplets by value
        i = indices[:, 0]
        j = indices[:, 1]
        k = indices[:, 2]
        idx = sum_vector.argsort()
        indices = np.dstack((i[idx], j[idx], k[idx]))

        # a = indices[0, 0, :]
        # b = indices[0, 1, :]
        # print(a)
        # print(b)
        # print(spans[a])
        # print(spans[b])

        # a = indices[0, 0, :]
        # length = norm_vector[idx[0]]
        # ortho = ortho_vector[idx[0]]
        # print(ortho)
        # print(length)
        # print(a)
        # print(spans[a])
        # print(ortho_vector[a])

        # Use the span with lowest score
        cell = spans[indices[0, 0, :]]

        return cell

    def _find_cell_atoms(self, system, seed_index, cell_basis_vectors):
        """Finds the atoms that are within the cell defined by the seed atom
        and the basis vectors.

        Args:
        Returns:

        """
        # Find the atoms within the found cell
        new_basis_inverse = np.linalg.inv(cell_basis_vectors.T)
        positions = system.get_positions()
        numbers = system.get_atomic_numbers()
        seed_pos = positions[seed_index]
        vec_new = np.dot(positions - seed_pos, new_basis_inverse.T)

        cell_pos = []
        cell_numbers = []
        a_prec, b_prec, c_prec = self.pos_tol/np.linalg.norm(cell_basis_vectors, axis=1)

        # If no positions are defined, find the atoms within the cell
        for i_pos, pos in enumerate(vec_new):
            x = 0 - a_prec <= pos[0] < 1 - a_prec
            y = 0 - b_prec <= pos[1] < 1 - b_prec
            z = 0 - c_prec <= pos[2] < 1 - c_prec
            if x and y and z:
                cell_pos.append(pos)
                cell_numbers.append(numbers[i_pos])

        return np.array(cell_pos), np.array(cell_numbers)

    def _get_repeated_system(self):
        """
        """
        if self._repeated_system is None:
            cell = self.system.get_cell()
            multiplier = np.ceil(self.max_dist/np.linalg.norm(cell, axis=1)
                ).astype(int)
            repeated = self.system.repeat(multiplier)
            self._repeated_system = repeated
        return self._repeated_system

    def _find_orthogonal_direction(self):
        """Used to find the unit vector that is orthogonal to the surface.

        Returns:
            The lattice vector in the original lattice that is
            orthogonal to the surface.

        Raises:
            ValueError: If the given system could not be identified as a
            surface.
        """
        # If necessary repeat the system in order to distinguish the surface
        # better
        if self._orthogonal_dir is None:
            repeated = self._get_repeated_system()

            # Get the eigenvalues and eigenvectors of the moment of inertia tensor
            val, vec = repeated.get_moments_of_inertia(vectors=True)
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

            # Find out the cell direction that corresponds to the orthogonal one
            cell = repeated.get_cell()
            dots = np.abs(np.dot(orthogonal_dir, cell.T))
            orthogonal_vector_index = np.argmax(dots)
            orthogonal_vector = cell[orthogonal_vector_index]
            self._orthogonal_dir = orthogonal_vector/np.linalg.norm(orthogonal_vector)

        return self._orthogonal_dir

    def _find_seed_points(self, system):
        """Used to find the given number of seed points where the symmetry
        search is started.

        The search is initiated from the middle of the system, and then
        additional seed points are added the direction orthogonal to the
        surface.
        """
        # from ase.visualize import view
        # view(system)

        orthogonal_dir = self._find_orthogonal_direction()

        # Determine the "width" of the system in the orthogonal direction
        positions = system.get_positions()
        components = np.dot(positions, orthogonal_dir)
        min_index = np.argmin(components)
        max_index = np.argmax(components)
        min_value = components[min_index]
        max_value = components[max_index]
        seed_range = np.linspace(min_value, max_value, 2*self.n_seeds-1)

        # Pick the initial surface position from the middle
        positions = system.get_positions()
        middle = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - middle, axis=1)
        init_seed_index = np.argmin(distances)
        init_seed_pos = positions[init_seed_index]
        ortho_init_comp = np.dot(init_seed_pos, orthogonal_dir)
        init_seed_pos = init_seed_pos - ortho_init_comp*orthogonal_dir

        # Find the seed atom closest to each seed point
        seed_indices = np.zeros(2*self.n_seeds-1, dtype=int)
        for i_point, point in enumerate(seed_range):
            seed_pos = init_seed_pos + point*orthogonal_dir
            distances = np.linalg.norm(positions - seed_pos, axis=1)
            seed_index = np.argmin(distances)
            seed_indices[i_point] = seed_index

        return seed_indices
