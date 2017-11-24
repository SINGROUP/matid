from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

import numpy as np

from ase import Atoms
from ase.visualize import view

import systax.geometry
from systax.core.linkedunits import LinkedUnitCollection, LinkedUnit


class PeriodicFinder():
    """Used to find translationally periodic structures within atomic systems.
    """
    def __init__(self, pos_tol, seed_algorithm, max_cell_size):
        """
        """
        self.pos_tol = pos_tol
        self.seed_algorithm = seed_algorithm
        self.max_cell_size = max_cell_size

    def get_regions(self, system, vacuum_dir):
        """Tries to find the periodic regions, like surfaces, in an atomic
        system.

        Args:
            system(ase.Atoms): The system from which to find the periodic
                regions.
            orig_indices(sequence of int): The original indices of the given system in the
                oiginal system
            vacuum_dir()
        """
        # Find the seed points
        if self.seed_algorithm == "cm":
            seed_points = [self._find_seed_cm(system)]
        # seed_points = self._find_seed_points(system)

        # Find possible bases for each seed point
        regions = []

        for seed_index in seed_points:
            possible_spans, neighbour_mask = self._find_possible_bases(system, seed_index)
            best_basis = self._find_best_basis(
                system,
                seed_index,
                possible_spans,
                neighbour_mask,
                vacuum_dir
            )

            n_spans = len(best_basis)
            if best_basis is None or n_spans == 0:
                return []

            # Find the atoms within the found cell
            if n_spans == 3:
                proto_cell = self._find_cell_atoms_3d(system, seed_index, best_basis)
                periodic_indices = [0, 1, 2]
            elif n_spans == 2:
                proto_cell, seed_position = self._find_cell_atoms_2d(system, seed_index, best_basis)
                periodic_indices = [0, 1]

            # view(proto_cell)
            # print(seed_position)

            # Find a region that is spanned by the found unit cell
            unit_collection = self._find_periodic_region(
                system,
                seed_index,
                proto_cell,
                seed_position,
                periodic_indices)

            i_indices = unit_collection.get_indices()
            # print(i_indices)
            rec = unit_collection.recreate_valid()
            # print(len(unit_collection))
            # view(rec)
            if len(i_indices) > 0:
                regions.append((i_indices, unit_collection, proto_cell))

        # Combine regions that are identical. The algorithm used here is
        # suboptimal but there are not so many regions here to compare.
        n_regions = len(regions)
        similarity = np.zeros((n_regions, n_regions))
        intersection_threshold = 0.9
        for i in range(n_regions):
            for j in range(n_regions):
                if j > i:
                    i_ind = set(regions[i][0])
                    j_ind = set(regions[j][0])
                    l_total = len(i_ind.union(j_ind))
                    l_intersection = i_ind.intersection(j_ind)
                    l_sim = l_intersection/l_total
                    similarity[i, j] = l_sim

        # Return only dissimilar regions
        dissimilar = []
        initial = set(range(n_regions))
        for i in range(n_regions):
            for j in initial:
                if j >= i:
                    l_sim = similarity[i, j]
                    if l_sim >= intersection_threshold:
                        initial.remove(i)
                    dissimilar.append(j)

        # See if the found cell is OK
        region_tuples = []
        for i_region in dissimilar:
            l_ind, l_coll, l_cell = regions[i_region]
            l_atoms = system[l_ind]
            region_tuple = (l_ind, l_coll, l_atoms, l_cell)
            region_tuples.append(region_tuple)

        return region_tuples

    def _find_possible_bases(self, system, seed_index):
        """Finds all the possible vectors that might span a cell.
        """
        # Calculate a displacement tensor that takes into account the
        # periodicity of the system
        pos = system.get_positions()
        pbc = system.get_pbc()
        cell = system.get_cell()
        disp_tensor = systax.geometry.get_displacement_tensor(pos, pos, pbc, cell)

        # If the search radius exceeds beyond the periodic boundaries, extend the system
        # Get the vectors that span from the seed to all other atoms
        # disp_tensor = syscache["disp_tensor"]
        seed_spans = disp_tensor[:, seed_index]
        atomic_numbers = system.get_atomic_numbers()

        # Find indices of atoms that are identical to seed atom
        seed_element = atomic_numbers[seed_index]
        identical_elem_mask = (atomic_numbers == seed_element)

        # Only keep spans that are smaller than the maximum vector length
        seed_span_lengths = np.linalg.norm(seed_spans, axis=1)
        distance_mask = (seed_span_lengths < self.max_cell_size)
        # syscache["neighbour_mask"] = distance_mask

        # Form a combined mask and filter spans with it
        combined_mask = (distance_mask) & (identical_elem_mask)
        combined_mask[seed_index] = False  # Ignore self
        bases = seed_spans[combined_mask]

        return bases, distance_mask

    def _find_best_basis(self, system, seed_index, possible_spans, neighbour_mask, vacuum_dir):
        """Check which spans in the given list actually are translational bases
        on the surface.

        In order to be a valid span, there has to be at least one repetition of
        this span for all atoms that are nearby the seed atom.
        """
        # system = syscache["system"]
        # disp_tensor = syscache["disp_tensor"]
        positions = system.get_positions()
        numbers = system.get_atomic_numbers()

        # Find how many of the neighbouring atoms have a periodic copy in the
        # found directions
        neighbour_pos = positions[neighbour_mask]
        neighbour_num = numbers[neighbour_mask]
        one_direction = np.empty((len(possible_spans)), dtype=int)
        two_directions = np.empty((len(possible_spans)), dtype=int)
        for i_span, span in enumerate(possible_spans):
            add_pos = neighbour_pos + span
            sub_pos = neighbour_pos - span
            add_indices = systax.geometry.get_matches(system, add_pos, neighbour_num, self.pos_tol)
            sub_indices = systax.geometry.get_matches(system, sub_pos, neighbour_num, self.pos_tol)

            n_one = 0
            n_two = 0
            for i_ind in range(len(add_indices)):
                i_add = add_indices[i_ind]
                i_sub = sub_indices[i_ind]
                if i_add is not None and i_sub is not None:
                    n_two += 1
                if i_add is not None or i_sub is not None:
                    n_one += 1
            one_direction[i_span] = n_one
            two_directions[i_span] = n_two

        # Get the spans that come from the periodicity if they are smaller than
        # the maximum cell size
        periodic_spans = system.get_cell()[~vacuum_dir]
        periodic_span_lengths = np.linalg.norm(periodic_spans, axis=1)
        periodic_spans = periodic_spans[periodic_span_lengths < self.max_cell_size]
        n_neighbours = len(neighbour_pos)
        n_periodic_spans = len(periodic_spans)
        if n_periodic_spans != 0:
            periodic_ones = n_neighbours*np.ones((n_periodic_spans))
            periodic_twos = n_neighbours*np.ones((n_neighbours))
            possible_spans = np.concatenate((possible_spans, periodic_spans), axis=0)
            one_direction = np.concatenate((one_direction, periodic_ones), axis=0)
            two_directions = np.concatenate((two_directions, periodic_twos), axis=0)

        # Find the directions with most one-directional matches
        span_lengths = np.linalg.norm(possible_spans, axis=1)
        normed_spans = possible_spans/span_lengths[:, np.newaxis]
        dots = np.inner(possible_spans, possible_spans)
        max_valid_neighbours = np.max(one_direction)
        max_span_indices = np.where(one_direction == max_valid_neighbours)
        v2 = normed_spans[max_span_indices]

        # Find the dimensionality of the space spanned by this set of vectors
        # print(v2)
        n_dim = self._find_space_dim(v2)

        # Find best valid combination of spans by looking at the number of
        # neighbours that are periodically repeated by the spans, the
        # orthogonality of the spans and the length of the spans
        span_indices = list(range(len(possible_spans)))
        combos = np.array(list(itertools.combinations(span_indices, n_dim)))

        # First sort by the number of periodic neighbours that are generated.
        # This way we first choose spans that create most periodic neighbours.
        periodicity_scores = np.zeros(len(combos))
        for i_dim in range(n_dim):
            i_combos = combos[:, i_dim]
            i_scores = two_directions[i_combos]
            periodicity_scores += i_scores
        periodicity_indices = np.argsort(-periodicity_scores)

        # print(two_directions)
        # print(periodicity_scores)

        # Iterate over the combos until a linearly independent combo is found
        # and stalemates have been resolved by checking the orthogonality and
        # vector lengths
        best_periodicity_score = 0
        best_score = float('inf')
        best_combo = None
        for index in periodicity_indices:

            combo = combos[index]
            i_per_score = periodicity_scores[index]

            # Check that the combination is linearly independent
            area_threshold = 0.1
            volume_threshold = 0.1
            if n_dim == 2:
                i = combo[0]
                j = combo[1]
                a_norm = normed_spans[i]
                b_norm = normed_spans[j]
                orthogonality = np.linalg.norm(np.cross(a_norm, b_norm))
                if orthogonality < area_threshold:
                    continue
                else:
                    ortho_score = abs(dots[i, j])
                    norm_score = span_lengths[i] + span_lengths[j]
                    i_score = ortho_score + norm_score

            elif n_dim == 3:
                i = combo[0]
                j = combo[1]
                k = combo[2]
                a_norm = normed_spans[i]
                b_norm = normed_spans[j]
                c_norm = normed_spans[k]
                orthogonality = np.dot(np.cross(a_norm, b_norm), c_norm)
                if orthogonality < volume_threshold:
                    continue
                else:
                    ortho_score = abs(dots[i, j]) + abs(dots[j, k]) + abs(dots[i, k])
                    norm_score = span_lengths[i] + span_lengths[j] + span_lengths[k]
                    i_score = ortho_score + norm_score

            # print(i_per_score)
            # print(i_score)

            if i_per_score >= best_periodicity_score:
                best_periodicity_score = i_per_score
                if i_score < best_score:
                    best_score = i_score
                    best_combo = combo
            else:
                if best_combo is not None:
                    break

        best_spans = possible_spans[best_combo]

        return best_spans

    def _find_space_dim(self, normed_spans):
        """Used to get the dimensionality of the space that is span by a set of
        vectors.
        """
        # Get  triplets of spans (combinations)
        span_indices = range(len(normed_spans))
        indices = np.array(list(itertools.combinations(span_indices, 3)))

        # Calculate the orthogonality tensor from the dot products of the
        # normalized spans
        dots = np.inner(normed_spans, normed_spans)
        dot1 = np.abs(dots[indices[:, 0], indices[:, 1]])
        dot2 = np.abs(dots[indices[:, 1], indices[:, 2]])
        dot3 = np.abs(dots[indices[:, 0], indices[:, 2]])
        ortho_vector = dot1 + dot2 + dot3

        # Sort the triplets by value
        i = indices[:, 0]
        j = indices[:, 1]
        k = indices[:, 2]
        idx = ortho_vector.argsort()
        indices = np.dstack((i[idx], j[idx], k[idx]))

        # Get the triplet span with lowest score
        cell = normed_spans[indices[0, 0, :]]

        # Check the dimensionality of this set
        a = cell[0, :]
        b = cell[1, :]
        c = cell[2, :]
        triple_product = np.dot(np.cross(a, b), c)

        dim = 3
        area_limit = 0.1
        volume_limit = 0.1
        if triple_product < volume_limit:
            dim = 2
            cross1 = np.linalg.norm(np.cross(a, b))
            cross2 = np.linalg.norm(np.cross(a, c))
            cross3 = np.linalg.norm(np.cross(b, c))

            if cross1 < area_limit and cross2 < area_limit and cross3 < area_limit:
                dim = 1

        return dim

    def _find_cell_atoms_3d(self, system, seed_index, cell_basis_vectors):
        """Finds the atoms that are within the cell defined by the seed atom
        and the basis vectors.

        Args:
        Returns:
            ASE.Atoms: System representing the found cell.

        """
        # Find the atoms within the found cell
        positions = system.get_positions()
        numbers = system.get_atomic_numbers()
        seed_pos = positions[seed_index]
        indices = systax.geometry.get_positions_within_basis(positions, cell_basis_vectors, seed_pos, self.pos_tol)
        cell_pos = positions[indices]
        cell_numbers = numbers[indices]

        trans_sys = Atoms(
            cell=cell_basis_vectors,
            positions=cell_pos,
            symbols=cell_numbers
        )

        return trans_sys

    def _find_cell_atoms_2d(self, system, seed_index, cell_basis_vectors):
        """Used to find a cell for 2D materials. The best basis that is found
        only contains two basis vector directions that have been deduced from
        translational symmetry. The third cell vector is determined by this
        function by looking at which atoms fullfill the periodic repetition in
        the neighbourhood of the seed atom and in the direction orthogonal to
        the found two vectors.

        Args:
            system(ASE.Atoms): Original system
            seed_index(int): Index of the seed atom in system
            cell_basis_vectors(np.ndarray): Two cell basis vectors that span
                the 2D system

        Returns:
            ASE.Atoms: System representing the found cell.
            np.ndarray: Position of the seed atom in the new cell.
        """
        # Find the atoms that are repeated with the cell
        pos = system.get_positions()
        num = system.get_atomic_numbers()
        seed_pos = pos[seed_index]

        # Create test basis that is used to find atoms that follow the
        # translation
        a = cell_basis_vectors[0]
        b = cell_basis_vectors[1]
        c = np.cross(a, b)
        c_norm = c/np.linalg.norm(c)
        c_test = 2*self.max_cell_size*c_norm
        test_basis = np.array((a, b, c_test))
        origin = seed_pos-0.5*c_test
        # print(origin)
        # print(test_basis)

        # Convert positions to this basis
        indices, cell_pos_rel = systax.geometry.get_positions_within_basis(
            system,
            test_basis,
            origin,
            self.pos_tol,
            [True, True, False]
        )
        seed_basis_index = np.where(indices == seed_index)[0][0]
        cell_pos_cart = systax.geometry.to_cartesian(test_basis, cell_pos_rel)
        seed_basis_pos_cart = np.array(cell_pos_cart[seed_basis_index])
        # print(cell_pos_cart)
        # print(origin)
        # print(test_basis)
        # print(cell_pos_cart)

        # View system
        # new_num = num[indices]
        # test_sys = Atoms(
            # cell=test_basis,
            # scaled_positions=cell_pos_rel,
            # symbols=new_num
        # )
        # view(test_sys)

        # TODO: Add a function that checks that periodic copies are not allowed
        # in the same cell, or that atoms just outside the edge of the cell are
        # not missed

        # TODO: For each atom in the cell, test that after subtraction and addition
        # in the lattice basis directions there is an identical copy

        # Determine the real cell thickness by getting the maximum and minimum
        # heights of the cell
        c_comp = cell_pos_rel[:, 2]
        max_index = np.argmax(c_comp)
        min_index = np.argmin(c_comp)
        pos_min_rel = np.array([0, 0, c_comp[min_index]])
        pos_max_rel = np.array([0, 0, c_comp[max_index]])
        pos_min_cart = systax.geometry.to_cartesian(test_basis, pos_min_rel)
        pos_max_cart = systax.geometry.to_cartesian(test_basis, pos_max_rel)
        c_new_cart = pos_max_cart-pos_min_cart

        # We demand a minimum size for the c-vector even if the system seems to
        # be purely 2-dimensional. This is done because the 3D-space cannot be
        # searched properly if one dimension is flat.
        c_size = np.linalg.norm(c_new_cart)
        min_size = 2*self.pos_tol
        if c_size < min_size:
            c_new_cart = min_size*c_norm
        offset_cart = (test_basis[2, :]-c_new_cart)/2

        new_basis = np.array(test_basis)
        new_basis[2, :] = c_new_cart

        # Create translated system
        new_num = num[indices]
        seed_basis_pos_cart -= offset_cart
        new_sys = Atoms(
            cell=new_basis,
            positions=cell_pos_cart - offset_cart,
            symbols=new_num
        )
        # view(new_sys)

        return new_sys, seed_basis_pos_cart

    def _find_seed_cm(self, system):
        """Finds the index of the seed point closest to the center of mass of
        the system.
        """
        cm = system.get_center_of_mass()
        positions = system.get_positions()
        distances = systax.geometry.get_distance_matrix(system.get_cell(), cm, positions)
        min_index = np.argmin(distances)

        return min_index

    def _find_periodic_region(self, system, seed_index, unit_cell, seed_position, periodic_indices):
        """Used to find atoms that are generated by a given unit cell and a
        given origin.

        Args:
            system(ASE.Atoms): Original system from which the periodic
                region is searched
            seed_index(int): Index of the atom from which the search is started
            unit_cell(ASE.Atoms): Repeating unit from which the searched region
                is composed of
            seed_position(np.ndarray): Cartesian position of the seed atom with
                respect to the unit cell origin.
            periodic_indices(sequence of int): Indices of the basis vectors
                that are periodic
            flat_indices(sequence of int): Indices of the basis vectors
                that are nearly zero vectors.
        """
        positions = system.get_positions()
        atomic_numbers = system.get_atomic_numbers()
        seed_pos = positions[seed_index]
        seed_number = atomic_numbers[seed_index]

        # Create a map between an atomic number and indices in the system
        number_to_index_map = {}
        number_to_pos_map = {}
        atomic_number_set = set(atomic_numbers)
        for number in atomic_number_set:
            number_indices = np.where(atomic_numbers == number)[0]
            number_to_index_map[number] = number_indices
            number_to_pos_map[number] = positions[number_indices]

        searched_coords = set()
        collection = LinkedUnitCollection(system)

        # view(system)

        self._find_region_rec(
            system,
            collection,
            number_to_index_map,
            number_to_pos_map,
            seed_index,
            seed_pos,
            seed_number,
            unit_cell,
            seed_position,
            searched_coords,
            (0, 0, 0),
            periodic_indices)

        return collection

    def _find_region_rec(
            self,
            system,
            collection,
            number_to_index_map,
            number_to_pos_map,
            seed_index,
            seed_pos,
            seed_atomic_number,
            unit_cell,
            seed_offset,
            searched_coords,
            index,
            periodic_indices):
        """
        Args:
            periodic_indices(sequence of int): The indices of the basis vectors
                that are periodic
            flat_indices(sequence of int): The indices of the basis vectors
                that are zero or almost zero.
        """
        # Check if this cell has already been searched
        # print(periodic_indices)
        if index in searched_coords:
            return
        else:
            searched_coords.add(index)

        cell_pos = unit_cell.get_scaled_positions()
        cell_num = unit_cell.get_atomic_numbers()

        # Find atoms within the cell
        cell_basis = unit_cell.get_cell()
        all_indices, test_pos_rel = systax.geometry.get_positions_within_basis(
            system,
            cell_basis,
            seed_pos-seed_offset,
            self.pos_tol/2.0)

        # print(all_indices)

        # If the seed atom was not found for this cell, end the search
        if seed_index is None:
            return

        # Find the the indices and position of the seed atoms of neighbouring
        # units.
        new_seed_indices = []
        new_seed_pos = []

        # Here we decide the directions to which the search is extended. The
        # directions depend on the directions that were found to be periodic
        # for the seed atom.
        n_periodic_dim = len(periodic_indices)
        if n_periodic_dim == 3:
            multipliers = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        elif n_periodic_dim == 2:
            multis = np.array(list(itertools.product((-1, 0, 1), repeat=2)))
            multipliers = np.zeros((multis.shape[0], multis.shape[1]+1))
            multipliers[:, periodic_indices] = multis
            multipliers = np.delete(multipliers, 4, 0)

        possible_seed_pos = number_to_pos_map[seed_atomic_number]
        possible_seed_indices = number_to_index_map[seed_atomic_number]

        for it, multiplier in enumerate(multipliers):
            disloc = np.sum(multiplier.T[:, np.newaxis]*cell_basis, axis=0)
            seed_guess = seed_pos + disloc
            i_seed_disp = possible_seed_pos - seed_guess
            i_seed_dist = np.linalg.norm(i_seed_disp, axis=1)
            i_index = np.where(i_seed_dist <= 2*self.pos_tol)  # multiplier of two because to neighbouring cells might be offset to opposite directions
            i_index = i_index[0]
            new_seed_indices.append(i_index)

            if len(i_index) != 0:
                i_seed_pos = possible_seed_pos[i_index[0]]
            else:
                i_seed_pos = seed_guess
            new_seed_pos.append(i_seed_pos)

        # Update the cell for this seed point. This adds more noise to the cell
        # basis vectors but allows us to track curved lattices. The indices 22,
        # 16, and 14 are the indices for the [1,0,0], [0,1,0] and [0, 0, 1]
        # multipliers.
        if n_periodic_dim == 3:
            i_cell = np.vstack((new_seed_pos[22], new_seed_pos[16], new_seed_pos[14]))
            i_cell = i_cell - seed_pos
        elif n_periodic_dim == 2:
            i_cell = np.array(cell_basis)
            i_cell[periodic_indices, :] = (new_seed_pos[6]-seed_pos, new_seed_pos[4]-seed_pos)

        if len(all_indices) != 0:
            new_sys = Atoms(
                cell=cell_basis,
                scaled_positions=test_pos_rel,
                symbols=system.get_atomic_numbers()[all_indices]
            )
            # view(new_sys)

            # Find the atom indices that match the atoms in the unit cell
            matches = systax.geometry.get_matches(
                new_sys,
                unit_cell.get_positions(),
                cell_num,
                self.pos_tol)
            if None in matches:
                print(new_sys.get_positions())
                print(unit_cell.get_positions())
                view(new_sys)
                print(matches)

            # Create the new LinkedUnit and add it to the collection representing
            # the surface
            for match in matches:
                if match is not None:
                    new_unit = LinkedUnit(index, seed_index, seed_pos, i_cell, all_indices, matches)
                    collection[index] = new_unit
                    break
            # print("==========")
            # print(index)
            # print(i_cell)
            # print(all_indices)

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

            a = multiplier[0] + index[0]
            b = multiplier[1] + index[1]
            c = multiplier[2] + index[2]

            # Update the cell shape
            new_cell = Atoms(
                cell=i_cell,
                scaled_positions=cell_pos,
                symbols=cell_num
            )
            self._find_region_rec(
                system,
                collection,
                number_to_index_map,
                number_to_pos_map,
                i_seed_index,
                i_seed_pos,
                seed_atomic_number,
                new_cell,
                seed_offset,
                searched_coords,
                (a, b, c),
                periodic_indices
            )
