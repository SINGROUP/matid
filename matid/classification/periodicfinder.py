from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

from collections import deque, defaultdict, OrderedDict

import numpy as np
from numpy.core.umath_tests import inner1d

import networkx as nx

import chronic

from sklearn.cluster import DBSCAN

from ase import Atoms
from ase.visualize import view
from ase.data import covalent_radii

import matid.geometry
from matid.data import constants
from matid.core.linkedunits import LinkedUnitCollection, LinkedUnit
from matid.exceptions import SystaxError


class PeriodicFinder():
    """Used to find translationally periodic structures within atomic systems.
    """
    def __init__(
            self,
            angle_tol=constants.ANGLE_TOL,
            pos_tol_scaling=constants.POS_TOL_SCALING,
            cell_size_tol=constants.CELL_SIZE_TOL,
            max_2d_cell_height=constants.MAX_2D_CELL_HEIGHT,
            max_2d_single_cell_size=constants.MAX_SINGLE_CELL_SIZE,
            chem_similarity_threshold=constants.CHEM_SIMILARITY_THRESHOLD
        ):
        """
        Args:
            angle_tol(float): The angle below which vectors in the cell basis are
                considered to be parallel.
            pos_tol_factor(float): The factor for multiplying the position
                tolerance when searching neighbouring cell seed atoms.
            cell_size_tol(float): The tolerance for cell sizes to be considered
                equal. Given relative to the smallest cell size.
        """
        self.angle_tol = angle_tol
        self.pos_tol_scaling = pos_tol_scaling
        self.cell_size_tol = cell_size_tol
        self.max_2d_cell_height = max_2d_cell_height
        self.max_2d_single_cell_size = max_2d_single_cell_size,
        self.chem_similarity_threshold = chem_similarity_threshold

    def get_region(
            self,
            system,
            seed_index,
            max_cell_size,
            pos_tol,
            delaunay_threshold=None,
            bond_threshold=None,
            disp_tensor_mic=None,
            disp_factors=None,
            disp_tensor_finite=None,
            dist_matrix_radii_mic=None,
        ):
        """Tries to find the periodic regions, like surfaces, in an atomic
        system.

        Args:
            system(ase.Atoms): The system from which to find the periodic
                regions.
            seed_index(int): The index of the atom from which the search is
                initiated.
            max_cell_size(float): The maximum size of cell basis vectors.
            pos_tol(float): The tolerance that is allowed in the search. Given
                as absolute value in angstroms.

        Returns:
            linkedunitcollection or None: A LinkedUnitCollection object representing
                the region or None if no region could be identified.
        """
        if delaunay_threshold is None:
            delaunay_threshold = constants.DELAUNAY_THRESHOLD
        if bond_threshold is None:
            bond_threshold = constants.BOND_THRESHOLD

        # If the distance information is not given, calculate it here.
        pbc = system.get_pbc()
        if disp_tensor_mic is None and disp_factors is None and disp_tensor_finite is None and dist_matrix_radii_mic is None:
            pos = system.get_positions()
            cell = system.get_cell()
            pbc = system.get_pbc()
            disp_tensor_finite = matid.geometry.get_displacement_tensor(pos, pos)
            if pbc.any():
                disp_tensor_mic, disp_factors = matid.geometry.get_displacement_tensor(
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

        self.disp_tensor_mic = disp_tensor_mic
        self.disp_tensor_finite = disp_tensor_finite
        self.disp_factors = disp_factors
        self.dist_matrix_radii_mic = dist_matrix_radii_mic
        self.pos_tol = pos_tol
        self.max_cell_size = max_cell_size
        region = None

        with chronic.Timer("find_possible_bases"):
            possible_spans, neighbour_mask, neighbour_factors = self._find_possible_bases(system, seed_index)
        with chronic.Timer("find_proto_cell"):
            proto_cell, offset, dim = self._find_proto_cell(
                system,
                seed_index,
                possible_spans,
                neighbour_mask,
                neighbour_factors,
                bond_threshold
            )

        # 1D is not handled
        if dim == 1 or proto_cell is None:
            return None

        # The indices of the periodic dimensions.
        periodic_indices = list(range(dim))

        # view(proto_cell)
        # print(proto_cell.get_cell())

        # Find a region that is spanned by the found unit cell

        with chronic.Timer("tracking"):
            unit_collection = self._find_periodic_region(
                system,
                dim == 2,
                delaunay_threshold,
                bond_threshold,
                seed_index,
                proto_cell,
                offset,
                periodic_indices,
            )

        with chronic.Timer("basis_indices"):
            i_indices = unit_collection.get_basis_indices()
            if len(i_indices) > 0:
                region = unit_collection
                region._pos_tol = pos_tol
                return region

    def _find_possible_bases(self, system, seed_index):
        """Finds all the possible vectors that might span a cell.

        Returns:
            np.ndarray: Array of found basis vectors
            np.ndarray: Mask that filters the neighbouring atoms
            np.ndarray: Factors that reveal in which periodic copy the
                neighbours are.
        """
        # Calculate a displacement tensor that takes into account the
        # periodicity of the system
        disp_tensor = self.disp_tensor_mic

        # If the search radius exceeds beyond the periodic boundaries, extend the system
        # Get the vectors that span from the seed to all other atoms
        seed_spans = disp_tensor[:, seed_index]
        atomic_numbers = system.get_atomic_numbers()

        # Find indices of atoms that are identical to seed atom
        seed_element = atomic_numbers[seed_index]
        identical_elem_mask = (atomic_numbers == seed_element)

        # Only keep spans that are smaller than the maximum vector length
        seed_span_lengths = np.linalg.norm(seed_spans, axis=1)
        distance_mask = (seed_span_lengths < self.max_cell_size)

        # Form a combined mask and filter spans with it
        combined_mask = (distance_mask) & (identical_elem_mask)
        combined_mask[seed_index] = False  # Ignore self
        bases = seed_spans[combined_mask]
        neighbour_factors = self.disp_factors[seed_index, distance_mask, :]

        return bases, distance_mask, neighbour_factors

    def _find_proto_cell(
            self,
            system,
            seed_index,
            possible_spans,
            neighbour_mask,
            neighbour_factors,
            bond_threshold
        ):
        """Used to find the best candidate for a unit cell basis that could
        generate a periodic region in the structure.

        Args:

        Returns:
            ase.Atoms: A system representing the best cell that was found
            np.ndarray: Position of the seed atom in the cell
        """
        positions = system.get_positions()
        numbers = system.get_atomic_numbers()

        # Find how many and which of the neighbouring atoms have a periodic
        # copy in the found directions
        with chronic.Timer("find_neighbour_copies"):
            neighbour_pos = positions[neighbour_mask]
            neighbour_num = numbers[neighbour_mask]
            neighbour_indices = np.where(neighbour_mask)[0]
            n_neighbours = len(neighbour_pos)
            n_spans = len(possible_spans)
            metric = np.empty((len(possible_spans)), dtype=int)
            adjacency_lists = []
            adjacency_lists_add = []
            adjacency_lists_sub = []
            neighbour_nodes = list(zip(neighbour_indices, neighbour_factors))

            for i_span, span in enumerate(possible_spans):

                # Calculate the scaled position tolerance
                i_pos_tol = np.full(n_neighbours, self.get_scaled_position_tolerance(span))

                i_adj_list = defaultdict(list)
                i_adj_list_add = defaultdict(list)
                i_adj_list_sub = defaultdict(list)
                add_pos = neighbour_pos + span
                sub_pos = neighbour_pos - span
                add_indices, _, _, add_factors = matid.geometry.get_matches(system, add_pos, neighbour_num, i_pos_tol)
                sub_indices, _, _, sub_factors = matid.geometry.get_matches(system, sub_pos, neighbour_num, i_pos_tol)

                n_metric = 0
                for i_neigh in range(n_neighbours):
                    i_add = add_indices[i_neigh]
                    i_sub = sub_indices[i_neigh]
                    i_add_factor = add_factors[i_neigh]
                    i_sub_factor = sub_factors[i_neigh]

                    if i_add is not None or i_sub is not None:
                        origin_factor = neighbour_factors[i_neigh]

                        if i_add is not None:
                            n_metric += 1
                            dest_factor = origin_factor + i_add_factor
                            i_adj_list[(neighbour_indices[i_neigh], tuple(origin_factor))].append((i_add, tuple(dest_factor)))
                            i_adj_list_add[(neighbour_indices[i_neigh], tuple(origin_factor))].append((i_add, tuple(dest_factor)))
                        if i_sub is not None:
                            n_metric += 1
                            dest_factor = origin_factor + i_sub_factor
                            i_adj_list[(neighbour_indices[i_neigh], tuple(origin_factor))].append((i_sub, tuple(dest_factor)))
                            i_adj_list_sub[(neighbour_indices[i_neigh], tuple(origin_factor))].append((i_sub, tuple(dest_factor)))

                metric[i_span] = n_metric
                adjacency_lists.append(i_adj_list)
                adjacency_lists_add.append(i_adj_list_add)
                adjacency_lists_sub.append(i_adj_list_sub)

        # Get the spans that come from the original cell basis if they are
        # smaller than the maximum cell size.
        with chronic.Timer("find_basis_spans"):
            periodic_spans = system.get_cell()
            periodic_span_lengths = np.linalg.norm(periodic_spans, axis=1)
            periodic_filter = periodic_span_lengths <= self.max_cell_size
            n_periodic_spans = periodic_filter.sum()
            if n_periodic_spans != 0:
                for i_per_span, per_span in enumerate(periodic_spans):
                    if periodic_filter[i_per_span]:

                        # Add the basis to the spans and add a full metric score
                        # for it.
                        possible_spans = np.concatenate((possible_spans, [per_span]), axis=0)
                        metric = np.concatenate((metric, [2*n_neighbours]), axis=0)

                        # Create the adjacency lists for the periodic span. There
                        # is full periodicity to neighbouring cells.
                        per_adjacency_list = defaultdict(list)
                        per_adjacency_list_add = defaultdict(list)
                        per_adjacency_list_sub = defaultdict(list)
                        i_factor = np.array((0, 0, 0))
                        i_factor[i_per_span] = 1
                        for i_neigh, neigh_factor in neighbour_nodes:

                            neigh_tuple = tuple(neigh_factor)
                            per_adjacency_list[(i_neigh, neigh_tuple)].append((i_neigh, tuple(neigh_factor + i_factor)))
                            per_adjacency_list[(i_neigh, neigh_tuple)].append((i_neigh, tuple(neigh_factor - i_factor)))
                            per_adjacency_list_add[(i_neigh, neigh_tuple)].append((i_neigh, tuple(neigh_factor + i_factor)))
                            per_adjacency_list_sub[(i_neigh, neigh_tuple)].append((i_neigh, tuple(neigh_factor - i_factor)))

                        adjacency_lists.append(per_adjacency_list)
                        adjacency_lists_add.append(per_adjacency_list_add)
                        adjacency_lists_sub.append(per_adjacency_list_sub)

        # Find the directions that repeat the neighbours above some preset
        # threshold. This is used to eliminate directions that are caused by
        # pure chance. The maximum score that a direction can get is
        # 2*n_neighbours. We specify that the score must be above 75% percent
        # of this maximum score to be considered a valid direction.
        valid_span_indices = np.where(metric >= 0.75*n_neighbours)[0]
        if len(valid_span_indices) == 0:
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

        # Get the adjacency lists corresponding to the best spans
        with chronic.Timer("form_adjacency"):
            best_adjacency_lists = []
            best_adjacency_lists_add = []
            best_adjacency_lists_sub = []
            for i_span in best_combo:
                original_span_index = valid_span_indices[i_span]
                i_adjacency_list = adjacency_lists[original_span_index]
                i_adjacency_list_add = adjacency_lists_add[original_span_index]
                i_adjacency_list_sub = adjacency_lists_sub[original_span_index]
                best_adjacency_lists.append(i_adjacency_list)
                best_adjacency_lists_add.append(i_adjacency_list_add)
                best_adjacency_lists_sub.append(i_adjacency_list_sub)

        # Create a full periodicity graph for the found basis
        with chronic.Timer("form_periodicity_graph"):
            periodicity_graph_pbc = None
            full_adjacency_list_pbc = defaultdict(list)
            for i_adj in best_adjacency_lists:
                for key, value in i_adj.items():
                    full_adjacency_list_pbc[key].extend(value)
            periodicity_graph_pbc = nx.Graph(full_adjacency_list_pbc)

        # Expand the graph by exploring the links of the atoms that are in
        # neighbouring cells
        with chronic.Timer("form_periodicity_graph"):
            node_conn = {}
            for cell_node, neigh_fact in zip(neighbour_indices, neighbour_factors):
                for node in periodicity_graph_pbc.nodes():
                    node_index, node_factor = node
                    if node_index == cell_node and node_factor == tuple(neigh_fact):
                        connections = periodicity_graph_pbc[node]
                        disp = []
                        ns = []
                        for neighbour_index, neighbour_factor in connections:
                            i_disp = tuple(np.array(neighbour_factor) - np.array(node_factor))
                            disp.append(i_disp)
                            ns.append(neighbour_index)
                        node_conn[cell_node] = [ns, disp]

        new_graph = periodicity_graph_pbc.copy()
        for node in periodicity_graph_pbc.nodes():
            node_index, node_factor = node
            if node_factor != (0, 0, 0):
                i_conn = node_conn.get(node_index)
                if i_conn is None:
                    continue
                for i_ind, i_factor in zip(i_conn[0], i_conn[1]):
                    new_node = (i_ind, tuple(np.array(node_factor) + np.array(i_factor)))
                    if new_node not in new_graph:
                        new_graph.add_node(new_node)
                    if not new_graph.has_edge(node, new_node):
                        new_graph.add_edge(node, new_node)
        periodicity_graph_pbc = new_graph

        # import matplotlib.pyplot as plt
        # plt.subplot(111)
        # pos = nx.spring_layout(periodicity_graph_pbc)
        # nx.draw_networkx_nodes(periodicity_graph_pbc, pos)
        # nx.draw_networkx_edges(periodicity_graph_pbc, pos)
        # data = periodicity_graph_pbc.nodes(data=True)
        # labels = {x[0]: x[0] for x in data}
        # nx.draw_networkx_labels(periodicity_graph_pbc, pos, labels, font_size=16)
        # plt.show()

        # Get all disconnected subgraphs in the periodicity graph that takes
        # periodic boundaries into account
        graphs = [periodicity_graph_pbc.subgraph(c) for c in nx.connected_components(periodicity_graph_pbc)]

        # Filter out the basis atoms by checking how many were found with
        # respect to the number of copies of the seed atom. This equals to
        # checking that the size of the subgraph is similar to the size of the
        # graph where the seed atom is. This is needed because the number of
        # edges is not always sufficient when a lot is happening on the
        # surface.
        subgraph_size = []
        target_size = None
        for graph in graphs:
            nodes = graph.nodes()
            indices = set([node[0] for node in nodes])
            n_nodes = len(nodes)
            if seed_index in indices:
                target_size = n_nodes
            subgraph_size.append(n_nodes)
        temp_graphs = []
        for i_graph, graph_size in enumerate(subgraph_size):

            # Corresponds to the check \omega_v > 0.5n_{seed} in the article.
            if graph_size > 0.5*target_size:
                temp_graphs.append(graphs[i_graph])

        graphs = temp_graphs

        # Eliminate subgraphs that do not have enough periodicity.
        valid_graphs = []
        neighbourhood_set = set([(x[0], tuple(x[1])) for x in neighbour_nodes])
        for graph in graphs:

            # The periodicity is measured by the average degree of the nodes.
            # Only the degree of the nodes that are in the neighbourhood are
            # taken into account.
            degrees = []
            for node in graph.nodes():
                if node in neighbourhood_set:
                    i_degree = graph.degree(node)
                    degrees.append(i_degree)
            mean_degree = np.array(degrees).mean()

            # Corresponds to the check \omega_c > 2(d-1) in the article.
            if mean_degree > (dim-1)*2:
                valid_graphs.append(graph)

        # If no valid graphs found, no region can be tracked.
        if len(valid_graphs) == 0:
            return None, None, None

        # Each subgraph represents a group of atoms that repeat periodically in
        # each cell. Here we calculate a mean position of these atoms in the
        # cell.
        seed_nodes = None
        seed_group_index = None
        group_data_pbc = {
            "num": [],
            "ind": [],
            "nodes": []
        }
        # Determine the indices, nodes and numbers for each valid subgraph.
        # index_set = set()
        for i_graph, graph in enumerate(valid_graphs):
            nodes = graph.nodes()
            node_indices = [node[0] for node in nodes]

            if seed_index in set(node_indices):
                seed_group_index = i_graph
                seed_nodes = nodes

            group_data_pbc["ind"].append(node_indices)
            group_data_pbc["nodes"].append(nodes)
            group_data_pbc["num"].append(numbers[node_indices][0])

        # If the seed atom is not in a valid graph, no region could be found.
        if seed_group_index is None:
            return None, None, None

        if n_spans == 3:
            proto_cell, offset = self._find_proto_cell_3d(
                seed_index,
                seed_nodes,
                best_combo,
                best_spans,
                system,
                group_data_pbc,
                seed_group_index,
                best_adjacency_lists_add,
                best_adjacency_lists_sub,
            )
        elif n_spans == 2:

            # The seed group index can get updated by the cell search
            proto_cell, offset, seed_group_index = self._find_proto_cell_2d(
                seed_index,
                seed_nodes,
                best_combo,
                best_spans,
                system,
                group_data_pbc,
                seed_group_index,
                best_adjacency_lists_add,
                best_adjacency_lists_sub,
            )

            if proto_cell is None:
                return None, None, None

        two_valid_spans = n_spans == 2
        if n_spans == 3:
            # If the max_cell_size is bigger than an interlayer distance
            # between two 2D sheets, then a wrong cell with a lot of vacuum
            # might get detected. Here we check that the dimensionality of the
            # found 3D cell is correct.
            try:
                dimensionality = matid.geometry.get_dimensionality(proto_cell, bond_threshold)
            except SystaxError:
                return None, None, None
            if dimensionality != 3:
                # If the cell has three unit vectors, but does not exhibit the
                # correct dimensionality, try if two of the cell vectors are
                # still OK.
                if dimensionality == 2:
                    a_thickness = matid.geometry.get_thickness(proto_cell, 0)
                    b_thickness = matid.geometry.get_thickness(proto_cell, 1)
                    c_thickness = matid.geometry.get_thickness(proto_cell, 2)
                    reduced_dimension = np.argmin([a_thickness, b_thickness, c_thickness])
                    proto_cell = matid.geometry.get_minimized_cell(proto_cell, reduced_dimension, 2*self.pos_tol)
                    i_pbc = [True, True, True]
                    i_pbc[reduced_dimension] = False
                    cell_mask = [True, True, True]
                    cell_mask[reduced_dimension] = False
                    proto_cell.set_pbc(i_pbc)
                    best_combo = best_combo[cell_mask]
                    best_spans = best_spans[cell_mask]
                    two_valid_spans = True
                    n_spans = 2
                else:
                    return None, None, None

        if two_valid_spans:

            # If the best 2D vectors consists only of the simulation basis cell
            # vectors, check that these vectors are below a predefined size.
            # Otherwise the cell cannot be accepted because there is not enough
            # statistics about the cell contents to distinguish outliers.
            if n_periodic_spans > 0:
                periodic_span_indices = valid_span_indices[-n_periodic_spans:]
                best_span_ind = valid_span_indices[best_combo]
                if set(best_span_ind).issubset(set(periodic_span_indices)):
                    cell_lens = np.linalg.norm(best_spans, axis=1)
                    if np.any(cell_lens > self.max_2d_single_cell_size):
                        return None, None, None

            # Check the dimensionality
            dimensionality, cluster_labels = matid.geometry.get_dimensionality(proto_cell, bond_threshold, return_clusters=True)
            if dimensionality is None:
                # If the original system has more than one cluster, the system
                # has multiple stacked 2D sheets with identical periodicity. In
                # this case the unit cell should only comprise of atoms in the
                # cluster where the seed atom is in.
                for i_index, i_cluster in enumerate(cluster_labels):
                    try:
                        seed_group_index = i_cluster.index(seed_group_index)
                    except ValueError:
                        pass
                    else:
                        cluster_indices = i_cluster
                        break
                proto_cell = proto_cell[cluster_indices]

                # Retry to get the dimensionality for the cell in which the
                # cluster where the seed atom is in has been separated.
                # view(proto_cell)
                dimensionality = matid.geometry.get_dimensionality(proto_cell, bond_threshold)
                if dimensionality is None:
                    return None, None, None
                else:
                    if dimensionality != 2:
                        return None, None, None
            else:
                if dimensionality != 2:
                    return None, None, None

            # Check the cell thickness. 2D materials that are thicker than a
            # specified threshold are accepted.
            proto_cell = matid.geometry.get_minimized_cell(proto_cell, 2, 2*self.pos_tol)
            offset = proto_cell.get_positions()[seed_group_index]
            thickness = matid.geometry.get_thickness(proto_cell, 2)
            if thickness > self.max_2d_cell_height:
                return None, None, None

        return proto_cell, offset, n_spans

    def _find_proto_cell_3d(
            self,
            seed_index,
            seed_nodes,
            best_span_indices,
            best_spans,
            system,
            group_data_pbc,
            seed_group_index,
            adjacency_add,
            adjacency_sub,
        ):
        """Given a cell shape, this function is used to return a fully
        populated prototype unit cell for 3D systems.

        Args:
            seed_index(int): Index of the seed atom in the original system
            seed_nodes(list): List of tuples containing the node index as first
                entry, and the 3D index of the cell repetition as second entry.
                E.g. (0, (-1, 1, 2))
            best_span_indices():
            best_spans(np.ndarray): A set of basis vectors for the cell as 3x3
                array.
            system(ase.Atoms): Original system
            group_data_pbc():
            seed_group_index():
            adjacency_add():
            adjacency_sub():

        Returns:
            unit_cell(ase.Atoms): The unit cell
            offset(np.ndarray): The cartesian offset of the seed atom in the
                cell.
        """
        # Keep one occurrence for each seed index

        # Find the seed positions copies that are within the neighbourhood
        orig_cell = system.get_cell()

        # Find the cells in which the copies of the seed atom are at the
        # origin. Here we are reusing information from the displacement tensor
        # and the factors that tell in which periodic cell copy the match was
        # found originally.
        cells = np.zeros((len(seed_nodes), 3, 3))
        for i_node, node in enumerate(seed_nodes):

            node_index = node[0]
            node_factor = node[1]

            # multiplier: The direction in which the cell basis is searched. +1 or -1.
            # node_factor: The cell index of the starting node.
            # i_factor: The factor of the matched atom.

            # Handle each basis
            for i_basis in range(3):
                a_final_neighbour = None
                a_add = adjacency_add[i_basis][node]
                a_sub = adjacency_sub[i_basis][node]

                if a_add:
                    a_add_neighbour, i_add_factor = a_add[0]
                    if a_add_neighbour != node_index:
                        a_final_neighbour = a_add_neighbour
                        i_factor = i_add_factor
                        multiplier = 1
                elif a_sub:
                    a_sub_neighbour, i_sub_factor = a_sub[0]
                    if a_sub_neighbour != node_index:
                        a_final_neighbour = a_sub_neighbour
                        i_factor = i_sub_factor
                        multiplier = -1

                if a_final_neighbour is not None:
                    # Old version
                    # a_correction = np.dot((-np.array(node_factor) + np.array(i_factor)), orig_cell)
                    # a = multiplier*self.disp_tensor_finite[a_final_neighbour, node_index, :] + a_correction

                    # New version
                    a_correction = np.dot((-np.array(node_factor) + np.array(i_factor)), orig_cell)
                    a = self.disp_tensor_finite[a_final_neighbour, node_index, :] + a_correction
                    a *= multiplier

                else:
                    a = best_spans[i_basis, :]

                cells[i_node, i_basis, :] = a

        # Find the relative positions of atoms inside the cell. If for too many
        # cells atoms are found that do not belong to the basis, then the found
        # unit cell is not correct and a cell should not be returned.
        orig_pos = system.get_positions()
        inside_nodes = []
        inside_pos = []
        index_cell_map = {}
        for i_node, cell in zip(seed_nodes, cells):
            i_seed, i_seed_factor = i_node
            seed_pos = orig_pos[i_seed]

            if i_seed in index_cell_map:
                i_indices, i_pos, i_factors = index_cell_map[i_seed]
            else:
                i_indices, i_pos, i_factors = matid.geometry.get_positions_within_basis(
                    system,
                    cell,
                    seed_pos,
                    1e-8,
                    -1e-8,
                )

                # Here we ensure that the seed atom is included in the cell
                # positions. The search might miss the seed atom which is
                # exactly at the border of the cell
                # if i_seed not in i_indices:
                    # i_indices = np.append(i_indices, [i_seed], axis=0)
                    # i_pos = np.append(i_pos, np.array([[0, 0, 0.5]]), axis=0)
                    # i_factors = np.append(i_factors, [[0, 0, 0]], axis=0)

                index_cell_map[i_seed] = (i_indices, i_pos, i_factors)

            # if (i_seed == 6):
            # print(i_indices)

            # Add the seed node factor
            final_factors = []
            for factor in i_factors:
                i_final_factor = tuple(np.array(i_seed_factor) + factor)
                final_factors.append(i_final_factor)

            # Create nodes (index+factor) that were found
            i_cell_nodes = list(zip(i_indices, final_factors))
            inside_nodes.append(OrderedDict(zip(i_cell_nodes, range(len(i_cell_nodes)))))
            inside_pos.append(i_pos)

        # For each node in a network, find the first relative position. Wrap
        # and average these positions to get a robust final estimate.
        # averaged_rel_pos = np.zeros((len(group_data_pbc["ind"]), 3))
        averaged_rel_pos = []
        averaged_rel_num = []
        new_group_index = None

        for i_group, nodes in enumerate(group_data_pbc["nodes"]):
            group_num = group_data_pbc["num"][i_group]
            scaled_pos = []
            for group_node in nodes:
                for cell_nodes, cell_positions in zip(inside_nodes, inside_pos):
                    if group_node in cell_nodes:
                        pos_index = cell_nodes[group_node]
                        pos = cell_positions[pos_index]
                        scaled_pos.append(pos)
                        break

            # The basis location corresponding to this group is only added is
            # at least one occurrence is found in a cell.
            if len(scaled_pos) != 0:
                scaled_pos = np.array(scaled_pos)

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
                averaged_rel_pos.append(group_avg)
                averaged_rel_num.append(group_num)

            if i_group == seed_group_index:
                new_group_index = len(averaged_rel_num) - 1
        seed_group_index = new_group_index

        averaged_rel_pos = np.array(averaged_rel_pos)

        proto_cell = Atoms(
            scaled_positions=averaged_rel_pos,
            symbols=averaged_rel_num,
            cell=best_spans,
            pbc=[True, True, True]
        )
        offset = proto_cell.get_positions()[seed_group_index]

        return proto_cell, offset

    def _find_proto_cell_2d(
            self,
            seed_index,
            seed_nodes,
            best_span_indices,
            best_spans,
            system,
            group_data_pbc,
            seed_group_index,
            adjacency_add,
            adjacency_sub,
        ):
        """Given a cell shape, this function is used to return a fully
        populated prototype unit cell for 2D systems.

        Args:
            seed_index(int): Index of the seed atom in the original system
            seed_nodes(list): List of tuples containing the node index as first
                entry, and the 3D index of the cell repetition as second entry.
                E.g. (0, (-1, 1, 2))
            best_span_indices():
            best_spans(np.ndarray): A set of basis vectors for the cell as 3x3
                array.
            system(ase.Atoms): Original system
            group_data_pbc():
            seed_group_index():
            adjacency_add():
            adjacency_sub():

        Returns:
            unit_cell(ase.Atoms): The unit cell
            offset(np.ndarray): The cartesian offset of the seed atom in the
                cell.
        """
        orig_cell = system.get_cell()

        # In 2D systems the maximum thickness of the system is defined by
        # max_cell_size.
        cutoff = self.max_cell_size

        # We need to make the third basis vector
        a = best_spans[0]
        b = best_spans[1]
        c = np.cross(a, b)
        c_norm = c/np.linalg.norm(c)
        c_norm = c_norm[None, :]
        c_test = 2*c_norm*cutoff

        basis = np.concatenate((best_spans, c_test), axis=0)

        # Find the cells in which the copies of the seed atom are at the
        # origin. Here we are reusing information from the displacement tensor
        # and the factors that tell in which periodic cell copy the match was
        # found originally.
        cells = np.zeros((len(seed_nodes), 3, 3))
        c_norms = np.zeros((len(seed_nodes), 3))
        for i_node, node in enumerate(seed_nodes):

            node_index = node[0]
            node_factor = node[1]

            # Handle each basis
            for i_basis in range(2):
                a_final_neighbour = None
                a_add = adjacency_add[i_basis][node]
                a_sub = adjacency_sub[i_basis][node]

                if a_add:
                    a_add_neighbour, i_add_factor = a_add[0]
                    if a_add_neighbour != node_index:
                        a_final_neighbour = a_add_neighbour
                        i_factor = i_add_factor
                        multiplier = 1
                elif a_sub:
                    a_sub_neighbour, i_sub_factor = a_sub[0]
                    if a_sub_neighbour != node_index:
                        a_final_neighbour = a_sub_neighbour
                        i_factor = i_sub_factor
                        multiplier = -1

                if a_final_neighbour is not None:
                    a_correction = np.dot((-np.array(node_factor) + np.array(i_factor)), orig_cell)
                    a = multiplier*self.disp_tensor_finite[a_final_neighbour, node_index, :] + a_correction
                else:
                    a = best_spans[i_basis, :]
                cells[i_node, i_basis, :] = a

            # Update the third axis for each cell.
            a = cells[i_node, 0]
            b = cells[i_node, 1]
            c = np.cross(a, b)
            c_norm = c/np.linalg.norm(c)
            c_norms[i_node, :] = c_norm
            c_norm = c_norm[None, :]
            c = 2*c_norm*cutoff
            cells[i_node, 2, :] = c

        # Find the relative positions of atoms inside the cell
        orig_pos = system.get_positions()
        inside_nodes = []
        inside_pos = []
        index_cell_map = {}

        for i_unit, (i_node, cell) in enumerate(zip(seed_nodes, cells)):
            i_seed, i_seed_factor = i_node
            seed_pos = orig_pos[i_seed]
            search_offset = -c_norms[i_unit, :]*cutoff
            search_coord = (seed_pos + search_offset)

            if i_seed in index_cell_map:
                i_indices, i_pos, i_factors = index_cell_map[i_seed]
            else:
                i_indices, i_pos, i_factors = matid.geometry.get_positions_within_basis(
                    system,
                    cell,
                    search_coord,
                    1e-8,
                    -1e-8,
                )

                # Here we ensure that the seed atom is included in the cell
                # positions. The search might miss the seed atom which is
                # exactly at the border of the cell
                # if i_seed not in i_indices:
                    # i_indices = np.append(i_indices, [i_seed], axis=0)
                    # i_pos = np.append(i_pos, np.array([[0, 0, 0.5]]), axis=0)
                    # i_factors = np.append(i_factors, [[0, 0, 0]], axis=0)

                index_cell_map[i_seed] = (i_indices, i_pos, i_factors)

            # Add the seed node factor
            final_factors = []
            for factor in i_factors:
                i_final_factor = tuple(np.array(i_seed_factor) + factor)
                final_factors.append(i_final_factor)

            # Create nodes (index+factor) that were found
            i_cell_nodes = list(zip(i_indices, final_factors))
            inside_nodes.append(OrderedDict(zip(i_cell_nodes, range(len(i_cell_nodes)))))
            inside_pos.append(i_pos)

        # For each node in a network, find the first relative position. Wrap
        # and average these positions to get a robust final estimate.
        averaged_rel_pos = []
        averaged_rel_num = []
        new_group_index = None
        for i_group, nodes in enumerate(group_data_pbc["nodes"]):
            scaled_pos = []
            group_num = group_data_pbc["num"][i_group]

            for group_node in nodes:
                for cell_nodes, cell_positions in zip(inside_nodes, inside_pos):
                    if group_node in cell_nodes:
                        pos_index = cell_nodes[group_node]
                        pos = cell_positions[pos_index]
                        scaled_pos.append(pos)
                        break

            # The basis location corresponding to this group is only added is
            # at least one occurrence is found in a cell.
            if len(scaled_pos) != 0:
                scaled_pos = np.array(scaled_pos)

                # Only wrap the 2D positions
                scaled_pos_2d = scaled_pos[:, 0:2]
                scaled_pos_2d %= 1

                # Find the copy with minimum distance from origin
                distances = np.linalg.norm(scaled_pos_2d, axis=1)
                min_dist_index = np.argmin(distances)
                min_dist_pos = scaled_pos_2d[min_dist_index]

                # All the other copies are moved periodically to be near the
                # position that is closest to origin.
                distances = scaled_pos_2d - min_dist_pos
                displacement = np.rint(distances)
                final_pos_2d = scaled_pos_2d - displacement

                # The average position is calculated
                scaled_pos[:, 0:2] = final_pos_2d
                group_avg = np.mean(scaled_pos, axis=0)
                averaged_rel_pos.append(group_avg)
                averaged_rel_num.append(group_num)

            if i_group == seed_group_index:
                new_group_index = len(averaged_rel_num) - 1
        seed_group_index = new_group_index

        # Move the seed positions back to the origin now that the search has
        # been performed
        averaged_rel_pos = np.array(averaged_rel_pos)

        proto_cell = Atoms(
            cell=basis,
            scaled_positions=averaged_rel_pos,
            symbols=averaged_rel_num,
            pbc=[True, True, False]
        )
        offset = proto_cell.get_positions()[seed_group_index]

        return proto_cell, offset, seed_group_index

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
        angle_thres_sin = abs(np.sin(angle_threshold))

        # Create combinations of normed spans
        span_indices = range(len(valid_spans))
        combo_indices = np.array(list(itertools.combinations(span_indices, 3)))
        normed_combos = norm_spans[combo_indices]

        # Create arrays containing the three angles for each combination. The
        # normalized zero vectors will have values NaN
        n_jk = np.cross(normed_combos[:, 1, :], normed_combos[:, 2, :])
        n_jk_norm = np.linalg.norm(n_jk, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            n_jk_hat = n_jk / n_jk_norm[:, None]
        alpha_ijk = np.abs(inner1d(normed_combos[:, 0, :], n_jk_hat))
        n_ki = np.cross(normed_combos[:, 2, :], normed_combos[:, 0, :])
        n_ki_norm = np.linalg.norm(n_ki, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            n_ki_hat = n_ki / n_ki_norm[:, None]
        alpha_jki = np.abs(inner1d(normed_combos[:, 1, :], n_ki_hat))
        n_ij = np.cross(normed_combos[:, 0, :], normed_combos[:, 1, :])
        n_ij_norm = np.linalg.norm(n_ij, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            n_ij_hat = n_ij / n_ij_norm[:, None]
        alpha_kij = np.abs(inner1d(normed_combos[:, 2, :], n_ij_hat))

        # The angles alpha, beta and gamma are cos(x), where x is the angle
        # between the vector and the normal of the plane. To get the angle y
        # between vector and plane, we use cos(x) = cos(pi/2 - y) = sin(y)
        with np.errstate(invalid='ignore'):
            ijk_mask = alpha_ijk >= angle_thres_sin
            jki_mask = alpha_jki >= angle_thres_sin
            kij_mask = alpha_kij >= angle_thres_sin
        angles_mask = ijk_mask & jki_mask & kij_mask

        # Number of valid angles for each combination
        n_valids = np.sum(angles_mask)

        # If there are three angles that are above the threshold, the cell is 3D
        if n_valids > 0:

            valid_indices = combo_indices[angles_mask]

            # Filter out combos that do not have metric score close to maximum
            # score that was found. This step is needed to filter out invalid
            metrics = valid_span_metrics[valid_indices]
            metric_sum = np.sum(metrics, axis=1)
            max_metric = metric_sum.max()
            metric_filter = metric_sum == max_metric
            valid_indices = valid_indices[metric_filter]

            # Filter the set into group with volume closest to the smallest
            # that was found
            set_norms = norms[valid_indices]
            set_volumes = alpha_ijk[angles_mask][metric_filter]*n_jk_norm[angles_mask][metric_filter]*np.prod(set_norms, axis=1)
            smallest_volume = set_volumes.min()
            smallest_cell_filter = set_volumes <= (1+self.cell_size_tol)*smallest_volume
            valid_indices = valid_indices[smallest_cell_filter]

            # From the group with smallest volume find a combination with
            # highest orthogonality
            n_ij_filtered = n_ij[angles_mask][metric_filter][smallest_cell_filter]
            n_ki_filtered = n_ki[angles_mask][metric_filter][smallest_cell_filter]
            n_jk_filtered = n_jk[angles_mask][metric_filter][smallest_cell_filter]
            ortho = 3 - inner1d(n_ij_filtered, n_ij_filtered) - inner1d(n_ki_filtered, n_ki_filtered) - inner1d(n_jk_filtered, n_jk_filtered)
            max_ortho_filter = np.argmin(ortho)
            best_span_indices = valid_indices[max_ortho_filter]

            # OLD VERSION
            # angle_sum = alpha_ijk[angles_mask] + alpha_jki[angles_mask] + alpha_kij[angles_mask]
            # angle_set = angle_sum[metric_filter][smallest_cell_filter]
            # biggest_angle_sum_filter = np.argmax(angle_set)
            # best_span_indices = valid_indices[biggest_angle_sum_filter]

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
        up_indices = np.triu_indices(n_spans, k=1)
        valid_indices_a = up_indices[0]
        valid_indices_b = up_indices[1]
        valid_indices = np.concatenate((valid_indices_a[:, None], valid_indices_b[:, None]), axis=1)
        crosses = pair_crosses[up_indices]
        sin_angle = np.abs(np.linalg.norm(crosses, axis=1))
        angle_filter = sin_angle >= angle_thres_sin
        valid_indices = valid_indices[angle_filter]
        n_cross_valids = len(valid_indices)

        # For 2D structures the best span pair has lowest area, angle
        # closest to 90 and sum of metric close to maximum that was found
        if n_cross_valids > 0:

            # Get all pairs that have metric close to maximum
            metrics = valid_span_metrics[valid_indices]
            metric_sum = np.sum(metrics, axis=1)
            max_metric = metric_sum.max()
            metric_filter = metric_sum == max_metric
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

    def _find_periodic_region(
            self,
            system,
            is_2d,
            tesselation_distance,
            bond_threshold,
            seed_index,
            unit_cell,
            seed_position,
            periodic_indices,
        ):
        """Used to find atoms that are generated by a given unit cell and a
        given origin.

        Args:
            system(ASE.Atoms): Original system from which the periodic
                region is searched
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

        searched_cell_indices = set()
        used_indices = set()
        used_seed_indices = set()
        searched_vacancy_positions = []
        queue = deque()
        collection = LinkedUnitCollection(
            system,
            unit_cell,
            is_2d,
            self.dist_matrix_radii_mic,
            self.disp_tensor_finite,
            tesselation_distance,
            self.chem_similarity_threshold,
            bond_threshold,
        )
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
            searched_cell_indices,
            (0, 0, 0),
            used_indices,
            searched_vacancy_positions,
            periodic_indices,
            queue,
            multipliers,
            used_seed_indices)

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
                    searched_cell_indices,
                    queue_index,
                    used_indices,
                    searched_vacancy_positions,
                    periodic_indices,
                    queue,
                    multipliers,
                    used_seed_indices)

        return collection

    def _get_multipliers(self, periodic_indices):
        """Used to calculate the multipliers that are used to multiply the cell
        basis vectors to find new unit cells.
        """
        # Here we decide the new seed points where the search is extended.
        n_periodic_dim = len(periodic_indices)
        if n_periodic_dim == 3:
            multipliers = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ])

        if n_periodic_dim == 2:
            multipliers = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
            ])

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
            searched_cell_indices,
            cell_index,
            used_indices,
            searched_vacancy_positions,
            periodic_indices,
            queue,
            multipliers,
            used_seed_indices
        ):
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
            searched_cell_indices(set): Set of 3D indices that have been searched.
            cell_index(tuple): The 3D coordinate of this unit cell.
            used_seed_indices(set): The indices that have been used as seeds.
            periodic_indices(sequence of int): The indices of the basis vectors
                that are periodic
        """
        # Check if this cell has already been searched
        if tuple(cell_index) in searched_cell_indices:
            return
        else:
            searched_cell_indices.add(tuple(cell_index))

        # Try to get the scaled positions for atoms in this new cell. If the
        # cell is non-invertible, then this cell is not processed.
        try:
            # Wrapping is here disabled because it does not handle well values
            # that are negative within machine precision.
            cell_pos = unit_cell.get_scaled_positions(wrap=False)
        except:
            return
        cell_num = unit_cell.get_atomic_numbers()
        old_basis = unit_cell.get_cell()

        new_seed_indices = []
        new_seed_pos = []
        new_cell_indices = []
        orig_cell = system.get_cell()
        orig_pbc = system.get_pbc()

        # Translate and wrap the searched positions
        test_pos = unit_cell.get_positions() - seed_offset + seed_pos
        test_pos = matid.geometry.to_scaled(orig_cell, test_pos, pbc=orig_pbc, wrap=True)
        test_pos = matid.geometry.to_cartesian(orig_cell, test_pos)

        # Find the atoms that match the positions in the original basis
        disps = unit_cell.get_positions() - seed_offset
        pos_tolerances = self.get_scaled_position_tolerance(disps)

        matches, substitutions, vacancies, _ = matid.geometry.get_matches(
            system,
            test_pos,
            cell_num,
            pos_tolerances,
        )

        # Associate the matched atoms to this cell
        for match in matches:
            if match is not None:
                collection._index_cell_map[match] = cell_index

        # Add all the matches into the lists containing already searched
        # locations.
        used_indices.update(matches)

        # Only allow substitutions that have not been added already.
        valid_substitutions = []
        new_subst = []
        for subst in substitutions:
            true_subst = None
            if subst is not None:
                subst_ind = subst.index
                if subst_ind not in used_indices:
                    true_subst = subst
                new_subst.append(subst_ind)
            valid_substitutions.append(true_subst)
        used_indices.update(new_subst)

        # Correct the vacancy positions by the seed pos, seed offset and cell
        # periodicity. Add the vacancy only if it has not already been added
        # before.
        new_vacancy_pos = []
        valid_vacancies = []
        vacancy_pos_array = np.array(searched_vacancy_positions)
        for vacancy in vacancies:
            # Check if this vacancy has already been found
            if len(searched_vacancy_positions) != 0:
                vac_dist = np.linalg.norm(vacancy_pos_array - vacancy.position, axis=1)
                if vac_dist.min() > self.pos_tol:
                    new_vacancy_pos.append(vacancy.position)
                    valid_vacancies.append(vacancy)
            else:
                new_vacancy_pos.append(vacancy.position)
                valid_vacancies.append(vacancy)
        searched_vacancy_positions.extend(new_vacancy_pos)

        # Find the neighbouring cells for extending the search
        dislocations = np.dot(multipliers, old_basis)
        new_cell, new_seed_indices, new_seed_pos, new_cell_indices = self._find_new_seeds_and_cell(
            system,
            seed_index,
            seed_pos,
            seed_atomic_number,
            dislocations,
            multipliers,
            old_basis,
            used_indices,
            cell_index,
            searched_cell_indices,
            used_seed_indices,
            collection._used_points,
            collection._search_graph,
            collection._index_cell_map
        )

        # If there are matches or substitutional atoms in the unit, add them to
        # the collection
        new_unit = LinkedUnit(cell_index, seed_index, seed_pos, new_cell, matches, valid_substitutions, valid_vacancies)
        collection[cell_index] = new_unit

        # Save the updated cell shape for the new cells in the queue
        new_sys = Atoms(
            cell=new_cell,
            scaled_positions=cell_pos,
            symbols=cell_num,
            pbc=unit_cell.get_pbc()
        )
        cells = len(new_seed_pos)*[new_sys]

        # Add the found neighbours to a queue
        queue.extend(list(zip(new_seed_indices, new_seed_pos, new_cell_indices, cells)))

    def get_scaled_position_tolerance(self, displacements):
        """Used to calculate the position tolerance that is scaled by the
        search distance.
        """
        # Add new axis to sigle vector displacements
        if len(displacements.shape) == 1:
            displacements = displacements[None, :]
        distance = np.linalg.norm(displacements, axis=1)
        scaled_tol = (1+self.pos_tol_scaling*distance)*self.pos_tol

        return scaled_tol

    def _find_new_seeds_and_cell(
            self,
            system,
            seed_index,
            seed_pos,
            seed_atomic_number,
            dislocations,
            multipliers,
            old_cell,
            used_indices,
            cell_index,
            searched_cell_indices,
            used_seed_indices,
            used_points,
            search_graph,
            index_cell_map
        ):
        """When given a prototype unit cell shape and a set of search
        directions, searches for new seed atoms that are used to initiate a
        search for a new repetition for a unit cell.

        Args:
            system(ase.Atoms): The system from which the seed atoms are
                searched.
            seed_index(int): The index of the atom from which the search is
                started.
            seed_pos(np.ndarray): The position vector of the seed atom.
            seed_atomic_number(int): The atomic number of the seed atom.
            dislocation(np.ndarray): An array of dislocation vectors given
                relative to the seed position.
            multipliers(np.ndarray): Multiplications of the unit cell
                corresponding to the given dislocation vectors.
            old_cell(np.ndarray): The unit cell given as 3x3 array.
            used_indices(set): A set of indices for atoms that have already
                been used as seed atoms or as part of unit cells in the system.
            cell_index(tuple): Index of the given cell in the
                LinkedUnitCollection. Given relative to the initial seed atom.
            searched_cell_indices(set of tuples): A set of cell indices that have
                already been searched.

        Returns:
            np.ndarray: The new unit cell that should be used when expanding
                the search.
            np.ndarray: Indices of the atoms that should be used as new seed
                atoms.
            np.ndarray: Positions of the new seed atoms.
            np.ndarray: Indices of the cells corresponding to the new seed
                atoms.
        """
        new_seed_indices = []
        new_seed_pos = []
        new_cell_indices = []
        new_cell = np.array(old_cell)

        # Check that no seed index is handled twice
        if seed_index in used_points:
            return new_cell, new_seed_indices, new_seed_pos, new_cell_indices
        else:
            used_points.add(seed_index)

        orig_cell = system.get_cell()
        orig_pos = system.get_positions()

        # Filter out cells that have already been searched
        test_cell_indices = multipliers + cell_index
        valid_multipliers = []
        for i_cell_ind, cell_ind in enumerate(test_cell_indices):
            # If the cell in this index has already been handled, continue
            if tuple(cell_ind) in searched_cell_indices:
                continue
            valid_multipliers.append(i_cell_ind)
        multipliers = multipliers[valid_multipliers]
        dislocations = dislocations[valid_multipliers]
        test_cell_indices = test_cell_indices[valid_multipliers]

        if seed_index is not None:

            # Find out the atoms that match the seed_guesses in the original
            # system
            seed_guesses = seed_pos + dislocations
            pos_tolerances = self.get_scaled_position_tolerance(dislocations)
            matches, _, _, factors = matid.geometry.get_matches(
                system,
                seed_guesses,
                len(dislocations)*[seed_atomic_number],
                pos_tolerances,
                mic=True
            )
            for match, factor, seed_guess, multiplier, disloc, test_cell_index in zip(
                    matches,
                    factors,
                    seed_guesses,
                    multipliers,
                    dislocations,
                    test_cell_indices):
                multiplier_tuple = tuple(multiplier)

                # Save the position corresponding to a seed atom or a guess for
                # it. If a match was found that is not the original seed, use
                # it's position to update the cell. If the matched index is the
                # same as the original seed, check the factors array to decide
                # whether to use the guess or not.
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

                # Check if this index has already been used as a seed. The
                # used_seed_indices is needed so that the same atom cannot
                # become a seed point multiple times. This can otherwise
                # become a problem in e.g. random systems, or "looped"
                # structures.
                add = True

                if match is not None:

                    # Get the index of the matched cell. If the matched atom is
                    # already associated with a cell, use that. Otherwise
                    # create a new index for the cell according to the
                    # multiplier.
                    if match in index_cell_map:
                        target_cell = index_cell_map[match]
                    else:
                        target_cell = cell_index + multiplier
                        index_cell_map[match] = target_cell

                    # Add an edge to the search graph
                    search_graph.add_node(tuple(cell_index), index=seed_index)
                    search_graph.add_node(tuple(target_cell), index=match)
                    search_graph.add_edge(tuple(cell_index), tuple(target_cell), multiplier=multiplier)

                    if match in used_indices:
                        add = False

                if add:
                    new_seed_indices.append(match)
                    new_seed_pos.append(i_seed_pos)
                    new_cell_indices.append(test_cell_index)

                    if match is not None:
                        used_indices.add(match)

                # Store the cell basis vector
                for i in range(3):
                    basis_mult = [0, 0, 0]
                    basis_mult[i] = 1
                    basis_mult = tuple(basis_mult)
                    if multiplier_tuple == basis_mult:
                        if match is None:
                            i_basis = disloc
                        else:
                            temp = i_seed_pos + np.dot(factor, orig_cell)
                            i_basis = temp - seed_pos
                        new_cell[i, :] = i_basis

        #TODO: Calculate the average cell for this seed atom. The average cell
        # is then used in the next phase of the search for the neighbouring
        # cells.

        return new_cell, new_seed_indices, new_seed_pos, new_cell_indices
