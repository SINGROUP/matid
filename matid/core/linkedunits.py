from collections import defaultdict, OrderedDict
import numpy as np

from ase import Atoms
from ase.data import covalent_radii
from ase.io import write

import matid.geometry
from matid.data import constants

import networkx as nx


class LinkedUnitCollection(dict):
    """Represents a collection of similar cells that are connected in 3D space
    to form a structure, e.g. a surface.

    Essentially this is a special flavor of a regular dictionary: the keys can
    only be a sequence of three integers, and the values should be LinkedUnits.
    """
    def __init__(
            self,
            system,
            cell,
            is_2d,
            dist_matrix_radii_pbc,
            disp_tensor_finite,
            delaunay_threshold=constants.DELAUNAY_THRESHOLD,
            chem_similarity_threshold=constants.CHEM_SIMILARITY_THRESHOLD,
            bond_threshold=constants.BOND_THRESHOLD,
        ):
        """
        Args:
            system(ase.Atoms): A reference to the system from which this
                LinkedUniCollection is gathered.
            cell(ase.Atoms): The prototype cell that is used in finding this
                region.
            is_2d(boolean): Whether this system represents a 2D-material or not.
            delaunay_threshold(float): The maximum allowed size of a tetrahedra
                edge in the Delaunay triangulation of the region..
        """
        self.system = system
        self.cell = cell
        self.is_2d = is_2d
        self.delaunay_threshold = delaunay_threshold
        self.chem_similarity_threshold = chem_similarity_threshold
        self.bond_threshold = bond_threshold
        self.dist_matrix_radii_pbc = dist_matrix_radii_pbc
        self.disp_tensor_finite = disp_tensor_finite
        self._search_graph = nx.MultiDiGraph()
        self._wrapped_moves = []
        self._index_cell_map = {}
        self._used_points = set()
        self._decomposition = None
        self._inside_indices = None
        self._outside_indices = None
        self._adsorbates = None
        self._substitutions = None
        self._vacancies = None
        self._clusters = None
        self._basis_indices = None
        self._basis_environments = None
        self._translations = None
        self._pos_tol = None
        dict.__init__(self)

    def __setitem__(self, key, value):
        # Transform key to tuple, check length
        try:
            key = tuple(key)
        except:
            raise TypeError(
                "Could not transform the given key '{}' into tuple."
                .format(key)
            )
        if len(key) != 3:
            raise ValueError(
                "The given coordinate '{}' does not have three components."
                .format(key)
            )

        # Check that old unit is not overwritten
        if key in dict.keys(self):
            raise ValueError(
                "Overriding existing units is not supported."
            )

        dict.__setitem__(self, key, value)

    def recreate_valid(self):
        """Used to recreate a new Atoms object, where each atom is created from
        a single unit cell. Atoms that were found not to belong to the periodic
        unit cell are not included.
        """
        recreated_system = Atoms(
            cell=self.system.get_cell(),
            pbc=self.system.get_pbc(),
        )
        for unit in self.values():
            i_valid_indices = np.array([x for x in unit.basis_indices if x is not None])
            if len(i_valid_indices) != 0:
                i_atoms = self.system[i_valid_indices]
                recreated_system += i_atoms

        return recreated_system


    def get_basis_atom_neighbourhood(self):
        """For each atom in the basis calculates the chemical neighbourhood.
        The chemical neighbourhood consists of a list of atomic numbers that
        are closer than a certain threshold value when the covalent radii is
        taken into account.

        Args:

        Returns:
        """
        if self._basis_environments is None:
            # Multiply the system to get the entire neighbourhood.
            cell = self.cell
            max_radii = covalent_radii[cell.get_atomic_numbers()].max()
            cutoff = max_radii + self.bond_threshold
            if self.is_2d:
                pbc = [True, True, False]
            else:
                pbc = [True, True, True]
            factors = matid.geometry.get_neighbour_cells(cell.get_cell(), cutoff, pbc)
            tvecs = np.dot(factors, cell.get_cell())

            # Find the factor corresponding to the original cell
            for i_factor, factor in enumerate(factors):
                if tuple(factor) == (0, 0, 0):
                    tvecs_reduced = np.delete(tvecs, i_factor, axis=0)
                    break

            pos = cell.get_positions()
            disp = matid.geometry.get_displacement_tensor(pos, pos, cell.get_cell())

            env_list = []
            for i in range(len(cell)):
                i_env = self.get_chemical_environment(
                    cell,
                    i,
                    disp,
                    tvecs,
                    tvecs_reduced
                )
                env_list.append(i_env)
            self._basis_environments = env_list

        return self._basis_environments

    def get_chem_env_translations(self):
        """Used to calculate the translations that are used in calculating the
        chemical enviroments.
        """
        if self._translations is None:
            cell = self.system.get_cell()
            num = self.system.get_atomic_numbers()
            max_radii = covalent_radii[num].max()
            cutoff = max_radii + self.bond_threshold
            factors = matid.geometry.get_neighbour_cells(cell, cutoff, True)
            translations = np.dot(factors, cell)

            # Find and remove the factor corresponding to the original cell
            for i_factor, factor in enumerate(factors):
                if tuple(factor) == (0, 0, 0):
                    translations_reduced = np.delete(translations, i_factor, axis=0)
                    break
            self._translations = (translations, translations_reduced)
        return self._translations

    def get_basis_indices(self):
        """Returns the indices of the atoms that were found to belong to a unit
        cell basis in the LinkedUnits in this collection as a single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        if self._basis_indices is None:
            # The chemical similarity check is completely skipped if threshold is zero
            if self.chem_similarity_threshold == 0:
                indices = set()
                for unit in self.values():
                    for index in unit.basis_indices:
                        if index is not None:
                            indices.add(index)
                self._basis_indices = indices
            else:
                translations, translations_reduced = self.get_chem_env_translations()

                # For each atom in the basis check the chemical environment
                neighbour_map = self.get_basis_atom_neighbourhood()

                indices = set()
                for unit in self.values():

                    # Compare the chemical environment near this atom to the one
                    # that is present in the prototype cell. If these
                    # neighbourhoods are too different, then the atom is not
                    # counted as being a part of the region.
                    for i_index, index in enumerate(unit.basis_indices):
                        if index is not None:
                            real_environment = self.get_chemical_environment(self.system, index, self.disp_tensor_finite, translations, translations_reduced)
                            ideal_environment = neighbour_map[i_index]
                            chem_similarity = self.get_chemical_similarity(ideal_environment, real_environment)
                            if chem_similarity >= self.chem_similarity_threshold:
                                indices.add(index)

                # Ensure that all the basis atoms belong to the same cluster.
                # clusters = self.get_clusters()
                self._basis_indices = np.array(list(indices))

        return self._basis_indices

    def get_outliers(self):
        """Returns the indices of atoms that were not found to belong to the
        basis.
        """
        basis_indices = set(self.get_basis_indices())
        all_indices = set(self.get_all_indices())
        additional_indices = np.array(list(all_indices - basis_indices))

        return additional_indices

    def get_chemical_environment(
            self,
            system,
            index,
            disp_tensor_finite,
            translations,
            translations_reduced
        ):
        """Get the chemical environment around an atom. The chemical
        environment is quantified simply by the number of different species
        around a certain distance when the covalent radii have been considered.
        """
        # Multiply the system to get the entire neighbourhood.
        num = system.get_atomic_numbers()
        n_atoms = len(system)
        seed_num = num[index]

        neighbours = defaultdict(lambda: 0)
        for j in range(n_atoms):
            j_num = num[j]
            ij_disp = disp_tensor_finite[index, j, :]

            if index == j:
                trans = translations_reduced
            else:
                trans = translations

            D_trans = trans + ij_disp
            D_trans_len = np.linalg.norm(D_trans, axis=1)
            ij_radii = covalent_radii[seed_num] + covalent_radii[j_num]
            ij_n_neigh = np.sum(D_trans_len - ij_radii <= self.bond_threshold)

            neighbours[j_num] += ij_n_neigh

        return neighbours

    def get_chemical_similarity(self, ideal_env, real_env):
        """Returns a metric that quantifies the similarity between two chemical
        environments. Here the metric is defined simply by the number
        neighbours that are found to be same as in the ideal environmen within
        a certain radius.
        """
        max_score = sum(ideal_env.values())

        score = 0
        for ideal_key, ideal_value in ideal_env.items():
            real_value = real_env.get(ideal_key)
            if real_value is not None:
                score += min(real_value, ideal_value)

        return score/max_score

    def get_interstitials(self):
        """Get the indices of interstitial atoms in the original system.
        """
        inside_indices, _ = self.get_inside_and_outside_indices()
        inside_indices = set(inside_indices)
        substitutions = self.get_substitutions()
        subst_indices = set()
        for subst in substitutions:
            subst_indices.add(subst.index)
        interstitials = inside_indices - subst_indices

        return np.array(list(interstitials))

    def get_clusters(self):
        """Used to cluster the system in order to distinguish chemisorbed
        adsorbates from the surface.
        """
        if self._clusters is None:
            clusters = matid.geometry.get_clusters(
                self.dist_matrix_radii_pbc,
                self.bond_threshold
            )
            clusters = [set(list(x)) for x in clusters]
            self._clusters = clusters

        return self._clusters

    def get_adsorbates(self):
        """Get the indices of the adsorbate atoms in the region.

        All atoms that are outside the tesselation, and are either not part of
        the elements present in the surface or further away than a certain
        threshold, are labeled as adsorbate atoms.

        This function does not differentiate between different adsorbate
        molecules.

        Returns:
            np.ndarray: Indices of the adsorbates in the original system.
        """
        if self._adsorbates is None:

            _, outside_indices = self.get_inside_and_outside_indices()
            basis_elements = set(self.cell.get_atomic_numbers())
            outside_indices = outside_indices

            # The substitutions have to be removed explicitly from the ouside
            # atoms because sometimes they are outside the tesselation.
            substitutions = self.get_substitutions()
            outside_indices = set(outside_indices)
            substitutions = set([x.index for x in substitutions])
            outside_indices -= substitutions
            outside_indices = np.array(list(outside_indices))

            if len(outside_indices) != 0:

                basis_elements = set(basis_elements)
                adsorbates = []
                for index in outside_indices:
                    adsorbates.append(index)
                adsorbates = np.array(adsorbates)
            else:
                adsorbates = np.array([])

            self._adsorbates = adsorbates

        return self._adsorbates

    def get_substitutions(self):
        """Get the substitutions in the region.
        """
        if self._substitutions is None:

            # Gather all substitutions
            # all_substitutions = []
            # for cell in self.values():
                # subst = cell.substitutions
                # if len(subst) != 0:
                    # all_substitutions.extend(subst)

            # The substitutions are validate based on their chemical
            # environment and position in the triangulation.
            neighbour_map = self.get_basis_atom_neighbourhood()
            valid_subst = []
            # _, outside_indices = self.get_inside_and_outside_indices()
            # outside_set = set(outside_indices)
            translations, translations_reduced = self.get_chem_env_translations()

            for unit in self.values():

                # Compare the chemical environment near this atom to the one
                # that is present in the prototype cell. If these
                # neighbourhoods are too different, then the atom is not
                # counted as being a part of the region.
                if unit.substitutions is not None:
                    for i_index, subst in enumerate(unit.substitutions):
                        if subst is not None:
                            subst_index = subst.index

                            # Otherwise check the chemical similarity
                            real_environment = self.get_chemical_environment(
                                self.system,
                                subst_index,
                                self.disp_tensor_finite,
                                translations,
                                translations_reduced
                            )
                            ideal_environment = neighbour_map[i_index]
                            chem_similarity = self.get_chemical_similarity(ideal_environment, real_environment)
                            if chem_similarity >= self.chem_similarity_threshold:
                                valid_subst.append(subst)
            self._substitutions = valid_subst

            # In 2D materials all substitutions in the cell are valid
            # substitutions
            # if self.is_2d:
                # self._substitutions = all_substitutions
            # else:
                # # In surfaces the substitutions have to be validate by whether they
                # # are inside the tesselation or not
                # inside_indices, _ = self.get_inside_and_outside_indices()
                # inside_set = set(inside_indices)

                # # Find substitutions that are inside the tesselation
                # valid_subst = []
                # for subst in all_substitutions:
                    # subst_index = subst.index
                    # if subst_index in inside_set:
                        # valid_subst.append(subst)
                # self._substitutions = valid_subst

        return self._substitutions

    def get_vacancies(self):
        """Get the vacancies in the region.

        Returns:
            ASE.Atoms: An atoms object representing the atoms that are missing.
            The Atoms object has the same properties as the original system.
        """
        if self._vacancies is None:

            # Gather all vacancies
            all_vacancies = []
            for cell in self.values():
                vacancies = cell.vacancies
                if len(vacancies) != 0:
                    all_vacancies.extend(vacancies)

            # For purely 2D systems all missing atoms in the basis are vacancies
            if self.is_2d:
                self._vacancies = all_vacancies
            else:
                # Get the tesselation
                tesselation = self.get_tetrahedra_decomposition()

                # Find substitutions that are inside the tesselation
                valid_vacancies = []
                for vacancy in all_vacancies:
                    vac_pos = vacancy.position
                    simplex = tesselation.find_simplex(vac_pos)
                    if simplex is not None:
                        valid_vacancies.append(vacancy)
                self._vacancies = valid_vacancies

        return self._vacancies

    def get_tetrahedra_decomposition(self):
        """Get the tetrahedra decomposition for this region.
        """
        if self._decomposition is None:
            # Get the positions of basis atoms
            basis_indices = self.get_basis_indices()
            valid_sys = self.system[basis_indices]

            # Perform tetrahedra decomposition
            self._decomposition = matid.geometry.get_tetrahedra_decomposition(
                valid_sys,
                self.delaunay_threshold
            )

        return self._decomposition

    def get_all_indices(self):
        """Get all the indices that are present in the full system.
        """
        return set(range(len(self.system)))

    def get_unknowns(self):
        """Returns indices of the atoms that are in the outliers but are not
        recognized as any specialized group.
        """
        outliers = set(self.get_outliers())
        adsorbates = set(self.get_adsorbates())
        interstitials = set(self.get_interstitials())
        substitutions = set([x.index for x in self.get_substitutions()])

        return outliers - adsorbates - interstitials - substitutions

    def get_inside_and_outside_indices(self):
        """Get the indices of atoms that are inside and outside the tetrahedra
        tesselation.

        Returns:
            (np.ndarray, np.ndarray): Indices of atoms that are inside and
            outside the tesselation. The inside indices are in the first array.
        """
        if self._inside_indices is None and self._outside_indices is None:
            invalid_indices = self.get_outliers()
            invalid_pos = self.system
            inside_indices = []
            outside_indices = []

            if len(invalid_indices) != 0:
                invalid_pos = self.system.get_positions()[invalid_indices]
                tesselation = self.get_tetrahedra_decomposition()
                for i, pos in zip(invalid_indices, invalid_pos):
                    simplex_index = tesselation.find_simplex(pos)
                    if simplex_index is None:
                        outside_indices.append(i)
                    else:
                        inside_indices.append(i)

            self._inside_indices = np.array(inside_indices)
            self._outside_indices = np.array(outside_indices)

        return self._inside_indices, self._outside_indices

    def get_connected_directions(self):
        """During the tracking of the region the information about searches
        that matched an atom twive but with a negated multiplier are stored.
        """
        connected_directions = np.array([False, False, False])

        # Find all the nodes that have at least two incoming edges. If there
        # are two incoming edges with negated multiplier, there is periodicity
        # in the multiplier direction.
        G = self._search_graph
        for node in G.nodes():
            node_edges = G.in_edges(node, data=True)
            multiplier_sum = np.array([0, 0, 0])
            multiplier_presence = np.array([False, False, False])
            for edge in node_edges:
                source = edge[0]
                dest = edge[1]
                unit_index_change = np.array(dest) - np.array(source)
                multiplier = edge[2]["multiplier"]
                multiplier_sum += multiplier

                # Create a mask that selects the index where the move occurred
                move_mask = multiplier != 0

                # If the movement does not correspond to the change in unit
                # cell index, then the movement has wrapped across and that
                # direction is periodic
                if multiplier[move_mask] != unit_index_change[move_mask]:
                    multiplier_presence[move_mask] = True

            for i in range(3):
                if multiplier_presence[i]:
                    if multiplier_sum[i] == 0:
                        connected_directions[i] = True

        # For each loop, we calculate the total displacement. If it is nonzero,
        # the nonzero direction is marked as a connected direction.
        # cycles = list(nx.simple_cycles(self._search_graph))
        # loop_multipliers = []
        # for i_loop, loop in enumerate(cycles):
            # loop_len = len(loop)

            # # A self-loop can be formed when the found vector corresponds to a
            # # simulation vector. In this case the loop has only one element,
            # # and all the edges are valid multipliers.
            # if loop_len == 1:
                # source = loop[0]
                # dest = loop[0]
                # edges = G[source][dest]
                # for key, edge in edges.items():
                    # multiplier = edge["multiplier"]
                    # loop_multipliers.append(multiplier)
            # # A loop between two atoms can be formed when there are only two
            # # repetitions of the cell per simulation cell basis vector. In any
            # # of the preset multipliers will be valid. Two-element loops within
            # # the simulation cell cannot be formed because the search never
            # # come back to an atom without crossing the periodic boundary.
            # elif loop_len == 2:
                # source = loop[0]
                # dest = loop[1]
                # edges = G[source][dest]
                # for key, edge in edges.items():
                    # multiplier = edge["multiplier"]
                    # if multiplier[0] != 0:
                        # print(multiplier)
                        # print(G.nodes[source])
                        # print(G.nodes[dest])
                    # loop_multipliers.append(multiplier)
            # # We can ignore loops with more elements. The two-body loops will
            # # already tell the directions in which the graph is periodic.
            # else:
                # pass

        # loop_multipliers = np.array(loop_multipliers)
        # indices = np.where(loop_multipliers != 0)[1]
        # indices = np.unique(indices)
        # connected_directions[indices] = True

        return connected_directions


class LinkedUnit():
    """Represents a cell that is connected to others in 3D space to form a
    structure, e.g. a surface.
    """
    def __init__(self, index, seed_index, seed_coordinate, cell, basis_indices, substitutions, vacancies):
        """
        Args:
            index(tuple of three ints):
            seed_index(int):
            seed_coordinate():
            cell(np.ndarray): Cell for this unit. Can change from unit to unit.
            basis_indices(sequence of ints and Nones): A sequence where there
                is an index or None for each atom that is supposed to be in the
                basis
            substitute_indices(sequence of ints and Nones): If basis atom is
                replaced by a foreign atom, the index of the substitutional atom is
                here.
        """
        self.index = index
        self.seed_index = seed_index
        self.seed_coordinate = seed_coordinate
        self.cell = cell
        self.basis_indices = basis_indices
        self.substitutions = substitutions
        self.vacancies = vacancies


class Substitution():
    """Represents a substitutional point defect.
    """
    def __init__(self, index, position, original_element, substitutional_element):
        """
        Args:
            index (int): Index of the subtitutional defect in the original
                system.
            original (ase.Atom): The original atom that was substituted
            substitution (ase.Atom): The substitutional atom

        """
        self.index = index
        self.original_element = original_element
        self.substitutional_element = substitutional_element
