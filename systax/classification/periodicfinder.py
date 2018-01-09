from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

from collections import deque, defaultdict

import numpy as np
from numpy.core.umath_tests import inner1d

import networkx as nx

from ase import Atoms
from ase.visualize import view

import systax.geometry
from systax.core.linkedunits import LinkedUnitCollection, LinkedUnit


class PeriodicFinder():
    """Used to find translationally periodic structures within atomic systems.
    """
    def __init__(
            self,
            pos_tol,
            angle_tol,
            max_cell_size,
            pos_tol_factor,
            cell_size_tol,
            n_edge_tol
        ):
        """
        Args:
            pos_tol(float): The positions tolerance for finding atoms in
                angstroms.
            angle_tol(float): The angle tolerance for separating parallel
                directions. Directions that have angle smaller than this value
                are considered to be parallel.
            max_cell_size(float): Guess for the maximum cell basis vector
                lengths. In angstroms.
            pos_tol_factor(float): A factor that is used to multiply the
                position tolerance when finding atoms in adjacent unit cell.
            cell_size_tol(float):
            n_edge_tol(float):
        """
        self.pos_tol = pos_tol
        self.angle_tol = angle_tol
        self.max_cell_size = max_cell_size
        self.pos_tol_factor = pos_tol_factor
        self.cell_size_tol = cell_size_tol
        self.n_edge_tol = n_edge_tol

    def get_region(self, system, seed_index, disp_tensor_pbc, vacuum_dir, tesselation_distance):
        """Tries to find the periodic regions, like surfaces, in an atomic
        system.

        Args:
            system(ase.Atoms): The system from which to find the periodic
                regions.
            seed_index(int): The index of the atom from which the search is
                initiated.
            vacuum_dir(sequence three booleans): The cell basis directions that
                have a vacuum gap.

        Returns:
            list of tuples: A list of tuples containing the following information:
                indices: Indices of the atoms belonging to a region
                linkedunitcollection: A LinkedUnitCollection object
                    representing the region
                unit cell: An ASE.Atoms object representing the unit cell of the region.
        """
        self.disp_tensor_pbc = disp_tensor_pbc
        region = None
        possible_spans, neighbour_mask = self._find_possible_bases(system, seed_index)
        proto_cell, offset, dim = self._find_proto_cell(
            system,
            seed_index,
            possible_spans,
            neighbour_mask,
            vacuum_dir
        )

        # 1D is not handled
        if dim == 1 or proto_cell is None:
            return None

        # The indices of the periodic dimensions.
        periodic_indices = list(range(dim))

        # view(proto_cell)
        # print(seed_position)

        # Find a region that is spanned by the found unit cell
        unit_collection = self._find_periodic_region(
            system,
            vacuum_dir,
            dim == 2,
            tesselation_distance,
            seed_index,
            proto_cell,
            offset,
            periodic_indices)

        i_indices = unit_collection.get_basis_indices()
        # rec = unit_collection.recreate_valid()
        # view(rec)

        if len(i_indices) > 0:
            region = (i_indices, unit_collection, proto_cell)

        return region

    def _find_possible_bases(self, system, seed_index):
        """Finds all the possible vectors that might span a cell.
        """
        # Calculate a displacement tensor that takes into account the
        # periodicity of the system
        disp_tensor = self.disp_tensor_pbc

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

    def _find_proto_cell(self, system, seed_index, possible_spans, neighbour_mask, vacuum_dir):
        """Used to find the best candidate for a unit cell basis that could
        generate a periodic region in the structure.

        Args:

        Returns:
            ase.Atoms: A system representing the best cell that was found
            np.ndarray: Position of the seed atom in the cell
        """
        vacuum_dir = np.array(vacuum_dir)
        positions = system.get_positions()
        numbers = system.get_atomic_numbers()

        # Find how many and which of the neighbouring atoms have a periodic
        # copy in the found directions
        neighbour_pos = positions[neighbour_mask]
        neighbour_num = numbers[neighbour_mask]
        neighbour_indices = np.where(neighbour_mask)[0]
        n_neighbours = len(neighbour_pos)
        n_spans = len(possible_spans)
        metric = np.empty((len(possible_spans)), dtype=int)
        metric_per_atom_per_span = np.zeros((n_neighbours, n_spans))
        adjacency_lists = []
        for i_span, span in enumerate(possible_spans):
            i_adj_list = defaultdict(list)
            add_pos = neighbour_pos + span
            sub_pos = neighbour_pos - span
            add_indices, _, _, _ = systax.geometry.get_matches(system, add_pos, neighbour_num, self.pos_tol)
            sub_indices, _, _, _ = systax.geometry.get_matches(system, sub_pos, neighbour_num, self.pos_tol)

            n_metric = 0
            for i_neigh in range(n_neighbours):
                i_add = add_indices[i_neigh]
                i_sub = sub_indices[i_neigh]
                if i_add is not None:
                    n_metric += 1
                    metric_per_atom_per_span[i_neigh, i_span] += 1
                    i_adj_list[neighbour_indices[i_neigh]].append(i_add)
                if i_sub is not None:
                    n_metric += 1
                    metric_per_atom_per_span[i_neigh, i_span] += 1
                    i_adj_list[neighbour_indices[i_neigh]].append(i_sub)
            metric[i_span] = n_metric
            adjacency_lists.append(i_adj_list)

        # Get the spans that come from the periodicity if they are smaller than
        # the maximum cell size
        periodic_spans = system.get_cell()[~vacuum_dir]
        periodic_span_lengths = np.linalg.norm(periodic_spans, axis=1)
        periodic_spans = periodic_spans[periodic_span_lengths < self.max_cell_size]
        n_periodic_spans = len(periodic_spans)
        if n_periodic_spans != 0:
            periodic_metric = 2*n_neighbours*np.ones((n_periodic_spans))
            possible_spans = np.concatenate((possible_spans, periodic_spans), axis=0)
            metric = np.concatenate((metric, periodic_metric), axis=0)
            for per_span in periodic_spans:
                per_adjacency_list = defaultdict(list)
                for i_neigh in neighbour_indices:
                    per_adjacency_list[i_neigh].extend([i_neigh, i_neigh])
                adjacency_lists.append(per_adjacency_list)

        # Find the directions that are most repeat the neighbours above some
        # preset threshold. This is used to eliminate directions that are
        # caused by pure chance. The maximum score that a direction can get is
        # 2*n_neighbours. We specify that the score must be above 25% percent
        # of this maximum score to be considered a valid direction.
        # span_lengths = np.linalg.norm(possible_spans, axis=1)
        # normed_spans = possible_spans/span_lengths[:, np.newaxis]
        # dots = np.inner(possible_spans, possible_spans)
        valid_span_indices = np.where(metric > 0.5*n_neighbours)
        if len(valid_span_indices[0]) == 0:
            return None, None, None

        # Find the best basis
        valid_span_metrics = metric[valid_span_indices]
        valid_spans = possible_spans[valid_span_indices]
        best_combo = self._find_best_basis(valid_spans, valid_span_metrics)
        dim = len(best_combo)

        # Currently 1D is not handled
        if dim == 1:
            return None, None, 1

        best_spans = valid_spans[best_combo]
        n_spans = len(best_spans)

        # Create a full periodicity graph for the found basis
        periodicity_graph = None
        full_adjacency_list = defaultdict(list)
        for ii_span, i_span in enumerate(best_combo):
            adjacency_list = adjacency_lists[i_span]
            for key, value in adjacency_list.items():
                full_adjacency_list[key].extend(value)
        periodicity_graph = nx.MultiGraph(full_adjacency_list)

        # import matplotlib.pyplot as plt
        # plt.subplot(111)
        # # nx.draw(periodicity_graph)
        # pos = nx.spring_layout(periodicity_graph)
        # nx.draw_networkx_nodes(periodicity_graph, pos)
        # data = periodicity_graph.nodes(data=True)
        # labels = {x[0]: x[0] for x in data}
        # # print(data)
        # nx.draw_networkx_labels(periodicity_graph, pos, labels, font_size=16)
        # plt.show()

        # Get all disconnected subgraphs
        graphs = list(nx.connected_component_subgraphs(periodicity_graph))

        # Eliminate subgraphs that do not have enough periodicity
        valid_graphs = []
        for graph in graphs:

            # The periodicity is measured by the average degree of the nodes.
            # The graph allows multiple edges, and edges that have the same
            # source and target due to periodicity.
            edges = graph.edges()
            node_edges = defaultdict(lambda: 0)
            for edge in edges:
                source = edge[0]
                target = edge[1]
                node_edges[source] += 1
                if source != target:
                    node_edges[target] += 1
            n_edges = np.array(list(node_edges.values()))
            mean_edges = n_edges.mean()

            if mean_edges >= dim:
                valid_graphs.append(graph)

        # If no valid graphs found, no region can be tracked.
        if len(valid_graphs) == 0:
            return None, None, None

        # Each subgraph represents a group of atoms that repeat periodically in
        # each cell. Here we calculate a mean position of these atoms in the
        # cell.
        seed_pos = positions[seed_index]

        group_pos = []
        group_num = []
        seed_group_index = None
        for i_graph, graph in enumerate(valid_graphs):
            nodes = graph.nodes(data=True)
            nodes = [node[0] for node in nodes]
            if seed_index in set(nodes):
                seed_group_index = i_graph
            nodes = np.array(nodes)
            graph_pos = positions[nodes]
            group_pos.append(graph_pos)
            group_num.append(numbers[nodes[0]])

        # If the seed atom is not in a valid graph, no region could be found.
        if seed_group_index is None:
            return None, None, None

        if n_spans == 3:
            proto_cell, offset = self._find_proto_cell_3d(best_spans, group_pos, group_num, seed_group_index, seed_pos)
        elif n_spans == 2:
            proto_cell, offset = self._find_proto_cell_2d(best_spans, group_pos, group_num, seed_group_index, seed_pos)

        return proto_cell, offset, n_spans

    def _find_proto_cell_3d(self, basis, pos, num, seed_index, seed_pos):
        """
        """
        # Each subgraph represents a group of atoms that repeat periodically in
        # each cell. Here we calculate a mean position of these atoms in the
        # cell.
        basis_element_positions = np.zeros((len(num), 3))
        basis_element_num = []
        for i_group, positions in enumerate(pos):

            # Calculate position in the relative basis of the found cell cell
            scaled_pos = systax.geometry.change_basis(positions, basis, seed_pos)
            scaled_pos %= 1

            # Find the copy with minimum distance from origin
            distances = np.linalg.norm(scaled_pos, axis=1)
            min_dist_index = np.argmin(distances)
            min_dist_pos = scaled_pos[min_dist_index]

            # All the other copies are moved periodically to be near the
            # position that is closest to origin.
            distances = scaled_pos - min_dist_pos
            displacement = np.rint(distances)
            final_pos = scaled_pos - displacement

            # The average position is calculated
            group_avg = np.mean(final_pos, axis=0)
            basis_element_positions[i_group] = group_avg
            basis_element_num.append(num[i_group])

        basis_element_num = np.array(basis_element_num)
        offset = basis_element_positions[seed_index]

        proto_cell = Atoms(
            scaled_positions=basis_element_positions,
            symbols=basis_element_num,
            cell=basis
        )

        return proto_cell, offset

    def _find_proto_cell_2d(self, basis, pos, num, seed_index, seed_pos):
        """
        Args:
            basis(np.ndarray): The basis cell vectors.
            pos(np.ndarray): A list of atomic positions. The list contains an
                entry containing multiple positions for each unique atom in the
                cell.
            num(np.ndarray): The atomic numbers of the cell atoms atoms.
            seed_index(int): The index of the seed atom in the pos and num arrays.
            seed_pos(np.ndarray): The position of the seed atom in cartesian
                coordinates.
        """
        # We need to make the third basis vector
        a = basis[0]
        b = basis[1]
        c = np.cross(a, b)
        c_norm = c/np.linalg.norm(c)
        c_norm = c_norm[None, :]
        basis = np.concatenate((basis, c_norm), axis=0)

        basis_element_positions = np.zeros((len(num), 3))
        basis_element_num = []
        for i_group, positions in enumerate(pos):

            # Calculate position in the relative basis of the found cell
            scaled_pos = systax.geometry.change_basis(positions, basis, seed_pos)
            scaled_pos_2d = scaled_pos[:, 0:2]
            scaled_pos_2d %= 1

            # Find the copy with minimum manhattan distance from origin
            distances = np.linalg.norm(scaled_pos_2d, axis=1)
            min_dist_index = np.argmin(distances)
            min_dist_pos = scaled_pos_2d[min_dist_index]

            # All the other copies are moved periodically to be near the min.
            # manhattan copy
            distances = scaled_pos_2d - min_dist_pos
            displacement = np.rint(distances)
            final_pos_2d = scaled_pos_2d - displacement

            # The average position is calculated
            scaled_pos[:, 0:2] = final_pos_2d
            group_avg = np.mean(scaled_pos, axis=0)
            basis_element_positions[i_group] = group_avg
            basis_element_num.append(num[i_group])

        # print(basis_element_positions)

        basis_element_num = np.array(basis_element_num)

        # Grow the cell to fit all atoms
        c_comp = basis_element_positions[:, 2]
        min_index = np.argmin(c_comp, axis=0)
        max_index = np.argmax(c_comp, axis=0)
        pos_min_rel = np.array([0, 0, c_comp[min_index]])
        pos_max_rel = np.array([0, 0, c_comp[max_index]])
        pos_min_cart = systax.geometry.to_cartesian(basis, pos_min_rel)
        pos_max_cart = systax.geometry.to_cartesian(basis, pos_max_rel)
        c_real_cart = pos_max_cart-pos_min_cart
        # print(c_new_cart)

        # We demand a minimum size for the c-vector even if the system seems to
        # be purely 2-dimensional. This is done because the 3D-space cannot be
        # searched properly if one dimension is flat.
        c_size = np.linalg.norm(c_real_cart)
        min_size = 2*self.pos_tol
        if c_size < min_size:
            c_inflated_cart = min_size*c_norm
            c_new_cart = c_inflated_cart
        else:
            c_new_cart = c_real_cart
        new_basis = np.array(basis)
        new_basis[2, :] = c_new_cart

        new_scaled_pos = basis_element_positions - pos_min_rel
        new_scaled_pos[:, 2] /= np.linalg.norm(c_new_cart)

        if c_size < min_size:
            offset_cart = (c_real_cart-c_inflated_cart)/2
            offset_rel = systax.geometry.to_scaled(new_basis, offset_cart)
            new_scaled_pos -= offset_rel

        # Create translated system
        # group_seed_pos = pos[seed_index]
        proto_cell = Atoms(
            cell=new_basis,
            scaled_positions=new_scaled_pos,
            symbols=basis_element_num
        )
        offset = proto_cell.get_positions()[seed_index]

        return proto_cell, offset

    def _find_best_basis(self, valid_spans, valid_span_metrics):
        """Used to choose the best basis from a set of valid ones.

        The given set of basis vectors should represent ones that reproduce the
        correct periodicity. This function then chooses one with correct
        dimensionality, minimal volume and maximal orthogonality.

        Args:
            valid_spans(np.ndarray):
            valid_span_metrics(np.ndarray):

        Returns:
            np.ndarray: Indices of the best spans from the given list of valid
            spans.
        """
        # Normed spans
        norms = np.linalg.norm(valid_spans, axis=1)
        norm_spans = valid_spans / norms[:, None]
        n_spans = len(valid_spans)

        if n_spans == 1:
            return [0]
        elif n_spans == 2:
            best_indices = self._find_best_2d_basis(norm_spans, norms, valid_span_metrics)
            return best_indices

        # The angle threshold for validating cells
        angle_threshold = np.pi/180*self.angle_tol
        angle_thres_sin = np.sin(angle_threshold)

        # Create combinations of normed spans
        span_indices = range(len(valid_spans))
        combo_indices = np.array(list(itertools.combinations(span_indices, 3)))
        normed_combos = norm_spans[combo_indices]

        # Create arrays containing the three angles for each combination. The
        # normalized zero vectors will have values NaN
        alpha_cross = np.cross(normed_combos[:, 1, :], normed_combos[:, 2, :])
        alpha_cross_norm = np.linalg.norm(alpha_cross, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha_cross_normed = alpha_cross / alpha_cross_norm[:, None]
        alphas = np.abs(inner1d(normed_combos[:, 0, :], alpha_cross_normed))
        beta_cross = np.cross(normed_combos[:, 2, :], normed_combos[:, 0, :])
        beta_cross_norm = np.linalg.norm(beta_cross, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            beta_cross_normed = beta_cross / beta_cross_norm[:, None]
        betas = np.abs(inner1d(normed_combos[:, 1, :], beta_cross_normed))
        gamma_cross = np.cross(normed_combos[:, 0, :], normed_combos[:, 1, :])
        gamma_cross_norm = np.linalg.norm(gamma_cross, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_cross_normed = gamma_cross / gamma_cross_norm[:, None]
        gammas = np.abs(inner1d(normed_combos[:, 2, :], gamma_cross_normed))

        with np.errstate(invalid='ignore'):
            alpha_mask = alphas > angle_thres_sin
            beta_mask = betas > angle_thres_sin
            gamma_mask = gammas > angle_thres_sin
        angles_mask = alpha_mask & beta_mask & gamma_mask

        # Number of valid angles for each combination
        n_valids = np.sum(angles_mask)

        # If there are three angles that are above the treshold, the cell is 3D
        if n_valids > 0:

            valid_indices = combo_indices[angles_mask]
            angle_sum = alphas[angles_mask] + betas[angles_mask] + gammas[angles_mask]

            # Filter out combos that do not have metric score close to maximum
            # score that was found. This step is needed to filter out invalid
            metrics = valid_span_metrics[valid_indices]
            metric_sum = np.sum(metrics, axis=1)
            max_metric = metric_sum.max()
            metric_filter = metric_sum > self.n_edge_tol*max_metric
            valid_indices = valid_indices[metric_filter]

            # Filter the set into group with volume closest to the smallest
            # that was found
            set_norms = norms[valid_indices]
            set_volumes = alphas[angles_mask][metric_filter]*alpha_cross_norm[angles_mask][metric_filter]*np.prod(set_norms, axis=1)
            smallest_volume = set_volumes.min()
            smallest_cell_filter = set_volumes < (1+self.cell_size_tol)*smallest_volume

            # From the group with smallest volume find a combination with
            # highest orthogonality
            angle_set = angle_sum[metric_filter][smallest_cell_filter]
            biggest_angle_sum_filter = np.argmax(angle_set)
            best_span_indices = valid_indices[smallest_cell_filter][biggest_angle_sum_filter]

        else:
            best_span_indices = self._find_best_2d_basis(norm_spans, norms, valid_span_metrics)

        return best_span_indices

    def _find_best_2d_basis(self, norm_spans, norms, valid_span_metrics):
        """Used to find the best 2D basis for a set of vectors.
        """
        # The angle threshold for validating cells
        angle_threshold = np.pi/180*self.angle_tol  # 10 degrees
        angle_thres_sin = np.sin(angle_threshold)

        # Calculate the cross-product between all normed pairs.
        n_spans = len(norm_spans)
        pair_crosses = np.zeros((n_spans, n_spans, 3))
        for i in range(n_spans):
            for j in range(n_spans):
                if j > i:
                    cross = np.cross(norm_spans[i], norm_spans[j])
                    pair_crosses[i, j, :] = cross
                    pair_crosses[j, i, :] = -cross

        # Get all pairs that have angle bigger than threshold
        up_indices = np.triu_indices(n_spans)
        valid_indices_a = up_indices[0]
        valid_indices_b = up_indices[1]
        valid_indices = np.concatenate((valid_indices_a[:, None], valid_indices_b[:, None]), axis=1)
        crosses = pair_crosses[up_indices]
        sin_angle = np.abs(np.linalg.norm(crosses, axis=1))
        angle_filter = sin_angle > angle_thres_sin
        valid_indices = valid_indices[angle_filter]
        n_cross_valids = len(valid_indices)

        # For 2D structures the best span pair has lowest area, angle
        # closest to 90 and sum of metric close to maximum that was found
        if n_cross_valids > 0:

            # Get all pairs that have metric close to maximum
            metrics = valid_span_metrics[valid_indices]
            metric_sum = np.sum(metrics, axis=1)
            # print(metric_sum)
            max_metric = metric_sum.max()
            metric_filter = metric_sum > self.n_edge_tol*max_metric
            valid_indices = valid_indices[metric_filter]

            # Find group of cells by finding cells with smallest area
            crosses = pair_crosses[valid_indices[:, 0], valid_indices[:, 1]]
            sin_angle = np.abs(np.linalg.norm(crosses, axis=1))
            valid_norms = np.prod(norms[valid_indices], axis=1)
            areas = valid_norms*sin_angle
            smallest_area = areas.min()
            smallest_cells_filter = areas < (1+self.cell_size_tol)*smallest_area
            valid_indices = valid_indices[smallest_cells_filter]

            # From the group with smallest area find a combination with
            # highest orthogonality
            crosses = pair_crosses[valid_indices[:, 0], valid_indices[:, 1]]
            sin_angle = np.abs(np.linalg.norm(crosses, axis=1))
            biggest_angle_index = np.argmax(sin_angle)

            best_span_indices = valid_indices[biggest_angle_index]

        # For 1D structures the best span is the shortest one
        else:
            best_span_indices = np.array([np.argmin(norms)])

        return best_span_indices

    def _find_periodic_region(self, system, vacuum_dir, is_2d,
            tesselation_distance, seed_index, unit_cell, seed_position,
            periodic_indices):
        """Used to find atoms that are generated by a given unit cell and a
        given origin.

        Args:
            system(ASE.Atoms): Original system from which the periodic
                region is searched
            vacuum_dir(sequence of boolean): The vacuum directions in the
                original simulation cell
            is_2d(boolean): Is the system a 2D material with cells in only one plane
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
        used_seed_indices = set()
        queue = deque()
        collection = LinkedUnitCollection(system, unit_cell, is_2d, vacuum_dir, tesselation_distance)
        multipliers = self._get_multipliers(periodic_indices)

        # Start off the queue
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
            used_seed_indices,
            periodic_indices,
            queue,
            multipliers)

        # Keep searching while new cells are found
        finished = False
        while not finished:
            try:
                queue_seed_index, queue_seed_pos, queue_index, queue_cell = queue.popleft()
            except IndexError:
                finished = True
            else:
                self._find_region_rec(
                    system,
                    collection,
                    number_to_index_map,
                    number_to_pos_map,
                    queue_seed_index,
                    queue_seed_pos,
                    seed_number,
                    queue_cell,
                    seed_position,
                    searched_coords,
                    queue_index,
                    used_seed_indices,
                    periodic_indices,
                    queue,
                    multipliers)

        return collection

    def _get_multipliers(self, periodic_indices):
        """Used to calculate the multipliers that are used to multiply the cell
        basis vectors to find new unit cells.
        """
        # Here we decide the new seed points where the search is extended. The
        # directions depend on the directions that were found to be periodic
        # for the seed atom.
        n_periodic_dim = len(periodic_indices)
        multipliers = []
        mult_gen = itertools.product((-1, 0, 1), repeat=n_periodic_dim)
        if n_periodic_dim == 2:
            for multiplier in mult_gen:
                if multiplier != (0, 0):
                    multipliers.append(multiplier)
        elif n_periodic_dim == 3:
            for multiplier in mult_gen:
                if multiplier != (0, 0, 0):
                    multipliers.append(multiplier)
        multipliers = np.array(multipliers)

        if n_periodic_dim == 2:
            multis = np.zeros((multipliers.shape[0], multipliers.shape[1]+1))
            multis[:, periodic_indices] = multipliers
            multipliers = multis

        return multipliers

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
            cell_index,
            used_seed_indices,
            periodic_indices,
            queue,
            multipliers):
        """
        Args:
            system(ASE.Atoms): The original system from which the periodic
                region is searched.
            collection(LinkedUnitCollection): Container for LinkedUnits that
                belong to a found region.
            number_to_index_map(dict): Connects atomic number to indices in the
                original system.
            number_to_pos_map(dict): Connects atomic number to positions in the
                original system.
            seed_index(int): Index of the seed atom in the original system.
            seed_pos(np.ndarray): Position of the seed atom in cartesian coordinates.
            seed_atomic_number(int): Atomic number of the seed atom.
            unit_cell(ASE.Atoms): The current guess for the unit cell.
            seed_offset(np.ndrray): Cartesian offset of the seed atom from the unit cell
                origin.
            searched_coords(set): Set of 3D indices that have been searched.
            cell_index(tuple): The 3D coordinate of this unit cell.
            used_seed_indices(set): The indices that have been used as seeds.
            periodic_indices(sequence of int): The indices of the basis vectors
                that are periodic
        """
        # Check if this cell has already been searched
        if tuple(cell_index) in searched_coords:
            return
        else:
            searched_coords.add(tuple(cell_index))

        # Try to get the scaled positions for atoms in this new cell. If the
        # cell is non-invertible, then this cell is not processed.
        try:
            cell_pos = unit_cell.get_scaled_positions()
        except:
            return
        cell_num = unit_cell.get_atomic_numbers()
        old_basis = unit_cell.get_cell()

        new_seed_indices = []
        new_seed_pos = []
        new_cell_indices = []
        orig_cell = system.get_cell()
        orig_pbc = system.get_pbc()

        # If the seed atom was not found for this cell, end the search
        dislocations = np.dot(multipliers, old_basis)
        new_cell, new_seed_indices, new_seed_pos, new_cell_indices = self._find_new_seeds_and_cell(
            system,
            seed_index,
            dislocations,
            multipliers,
            old_basis,
            used_seed_indices,
            cell_index,
            searched_coords)

        # Translate the original system to the seed position
        match_system = system.copy()
        match_system.translate(-seed_pos+seed_offset)

        # Find the atoms that match the positions in the original basis
        matches, substitutions, vacancies, _ = systax.geometry.get_matches(
            match_system,
            unit_cell.get_positions(),
            cell_num,
            self.pos_tol_factor*self.pos_tol,
            )

        # Correct the vacancy positions by the seed pos and cell periodicity
        for vacancy in vacancies:
            old_vac_pos = vacancy.position
            old_vac_pos += seed_pos
            vacancy_pos_rel = systax.geometry.to_scaled(orig_cell, old_vac_pos, pbc=orig_pbc, wrap=True)
            new_vac_pos = systax.geometry.to_cartesian(orig_cell, vacancy_pos_rel)
            vacancy.position = new_vac_pos

        # If there are maches or substitutional atoms in the unit, add it to
        # the collection
        new_unit = LinkedUnit(cell_index, seed_index, seed_pos, new_cell, matches, substitutions, vacancies)
        collection[cell_index] = new_unit

        # Save a snapshot of the process
        # from ase.io import write
        # rec = collection.recreate_valid()
        # rec.set_cell(system.get_cell())
        # num = len(collection)
        # str_num = str(num)
        # str_len = len(str_num)
        # num = (3-str_len)*"0" + str_num
        # write('/home/lauri/Desktop/2d/image_{}.png'.format(num), rec, rotation='-90x,45y,45x', show_unit_cell=2)
        # write('/home/lauri/Desktop/2d/image_{}.png'.format(num), rec, rotation='', show_unit_cell=2)
        # write('/home/lauri/Desktop/curved/image_{}.png'.format(num), rec, rotation='-80x', show_unit_cell=2)
        # write('/home/lauri/Desktop/crystal/image_{}.png'.format(num), rec, rotation='20y,20x', show_unit_cell=2)

        # Save the updated cell shape for the new cells in the queue
        new_sys = Atoms(
            cell=new_cell,
            scaled_positions=cell_pos,
            symbols=cell_num
        )
        cells = len(new_seed_pos)*[new_sys]

        # Add the found neighbours to a queue
        queue.extend(list(zip(new_seed_indices, new_seed_pos, new_cell_indices, cells)))

    def _find_new_seeds_and_cell(
            self,
            system,
            seed_index,
            dislocations,
            multipliers,
            old_cell,
            used_seed_indices,
            cell_index,
            searched_coords,
        ):
        """
        """
        orig_cell = system.get_cell()
        orig_pos = system.get_positions()
        orig_num = system.get_atomic_numbers()
        seed_pos = orig_pos[seed_index]
        seed_atomic_number = orig_num[seed_index]

        new_seed_indices = []
        new_seed_pos = []
        new_cell_indices = []
        new_cell = np.array(old_cell)

        # Filter out cells that have already been searched
        test_cell_indices = multipliers + cell_index
        valid_multipliers = []
        for i_cell_ind, cell_ind in enumerate(test_cell_indices):
            # If the cell in this index has already been handled, continue
            if tuple(cell_ind) in searched_coords:
                continue
            valid_multipliers.append(i_cell_ind)
        multipliers = multipliers[valid_multipliers]
        dislocations = dislocations[valid_multipliers]
        test_cell_indices = test_cell_indices[valid_multipliers]

        if seed_index is not None:

            # Find out the atoms that match the seed_guesses in the original
            # system
            seed_guesses = seed_pos + dislocations
            matches, _, _, factors = systax.geometry.get_matches(
                system,
                seed_guesses,
                len(dislocations)*[seed_atomic_number],
                self.pos_tol_factor*self.pos_tol)

            for match, factor, seed_guess, multiplier, disloc, test_cell_index in zip(
                    matches,
                    factors,
                    seed_guesses,
                    multipliers,
                    dislocations,
                    test_cell_indices):
                multiplier = tuple(multiplier)

                # Save the position corresponding to a seed atom or a guess for it.
                # If a match was found that is not the original seed, use it's
                # position to update the cell. If the matched index is the same as
                # the original seed, check the factors array to decide whether to
                # use the guess or not.
                if match is not None:
                    if match != seed_index:
                        i_seed_pos = orig_pos[match]
                    else:
                        if (factor == 0).all():
                            i_seed_pos = seed_guess
                        else:
                            i_seed_pos = orig_pos[match]
                else:
                    i_seed_pos = seed_guess

                # Store the indices and positions of new valid seeds
                if (factor == 0).all():
                    # Check if this index has already been used as a seed. The
                    # used_seed_indices is needed so that the same atom cannot
                    # become a seed point multiple times. This can otherwise
                    # become a problem in e.g. random systems, or "looped"
                    # structures.
                    add = True
                    if match is not None:
                        if match in used_seed_indices:
                            add = False
                    if add:
                        new_seed_indices.append(match)
                        new_seed_pos.append(i_seed_pos)
                        new_cell_indices.append(test_cell_index)
                        if match is not None:
                            used_seed_indices.add(match)

                # Store the cell basis vector
                for i in range(3):
                    basis_mult = [0, 0, 0]
                    basis_mult[i] = 1
                    basis_mult = tuple(basis_mult)
                    if multiplier == basis_mult:
                        if match is None:
                            i_basis = disloc
                        else:
                            temp = i_seed_pos + np.dot(factor, orig_cell)
                            i_basis = temp - seed_pos
                        new_cell[i, :] = i_basis

        return new_cell, new_seed_indices, new_seed_pos, new_cell_indices
