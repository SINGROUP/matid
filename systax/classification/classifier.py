from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


import itertools

import numpy as np

from ase.data import covalent_radii
from ase import Atoms
from ase.visualize import view

from systax.exceptions import ClassificationError
from systax.classification.components import SurfaceComponent, AtomComponent, MoleculeComponent, CrystalComponent, Material1DComponent, Material2DComponent, UnknownComponent
from systax.classification.classifications import Surface, Atom, Molecule, Crystal, Material1D, Material2D, Unknown
from systax.data.element_data import get_covalent_radii
import systax.geometry
from systax.analysis.material3danalyzer import Material3DAnalyzer
from systax.core.linkedunits import LinkedUnitCollection, LinkedUnit
from systax.symmetry import check_if_crystal
from systax.core.system import System


class SystemCache(dict):
    pass


class Classifier():
    """A class that is used to analyze the contents of an atomistic system and
    separate the consituent atoms into different components along with some
    meaningful additional information.
    """
    def __init__(
            self,
            max_cell_size=3,
            pos_tol=0.5,
            n_seeds=2,
            crystallinity_threshold=0.1,
            connectivity_crystal=1.9
            ):
        """
        Args:
            max_cell_size(float): The maximum cell size
            pos_tol(float): The position tolerance in angstroms for finding translationally
                repeated units.
            n_seeds(int): The number of seed positions to check.
            crystallinity_threshold(float): The threshold of number of symmetry
                operations per atoms in primitive cell that is required for
                crystals.
            connectivity_crystal(float): A parameter that controls the
                connectivity that is required for the atoms of a crystal.
        """
        self.max_cell_size = max_cell_size
        self.pos_tol = pos_tol
        self.n_seeds = 1
        self.crystallinity_threshold = crystallinity_threshold
        self.connectivity_crystal = connectivity_crystal
        self._repeated_system = None
        self._analyzed = False
        self._orthogonal_dir = None
        self.decisions = {}

    def classify(self, system):
        """A function that analyzes the system and breaks it into different
        components.

        Args:
            system(ASE.Atoms or System): Atomic system to classify.
        """
        self.system = system
        surface_comps = []
        atom_comps = []
        molecule_comps = []
        crystal_comps = []
        unknown_comps = []
        material1d_comps = []
        material2d_comps = []

        # Run a high level analysis to determine the system type
        periodicity = system.get_pbc()
        n_periodic = np.sum(periodicity)

        # Calculate a ratio of occupied space/cell volume. This ratio can
        # already be used to separate most bulk materials avoiding more
        # costly operations.
        try:
            cell_volume = system.get_volume()
        # If vectors are zero, volume not defined
        except ValueError:
            pass
        else:
            if cell_volume != 0:
                atomic_numbers = system.get_atomic_numbers()
                radii = covalent_radii[atomic_numbers]
                occupied_volume = np.sum(4.0/3.0*np.pi*radii**3)
                ratio = occupied_volume/cell_volume
                if ratio >= 0.3 and n_periodic == 3:
                    crystal_comps.append(AtomComponent(range(len(system)), system))
                    return Crystal(crystals=crystal_comps)

        if n_periodic == 0:

            # Check if system has only one atom
            n_atoms = len(system)
            if n_atoms == 1:
                atom_comps.append(AtomComponent([0], system))
            # Check if the the system has one or multiple components
            else:
                clusters = systax.geometry.get_clusters(system)
                for cluster in clusters:
                    n_atoms_cluster = len(cluster)
                    if n_atoms_cluster == 1:
                        atom_comps.append(AtomComponent(cluster, system[cluster]))
                    elif n_atoms_cluster > 1:
                        molecule_comps.append(MoleculeComponent(cluster, system[cluster]))

        # If the system has at least one periodic dimension, check the periodic
        # directions for a vacuum gap.
        else:

            # Find out the the eigenvectors and eigenvalues of the inertia
            # tensor for an extended version of this system.
            # extended_system = geometry.get_extended_system(system, 15)
            # eigval, eigvec = geometry.get_moments_of_inertia(extended_system)
            # print(eigval)
            # print(eigvec)
            vacuum_dir = systax.geometry.find_vacuum_directions(system)
            n_vacuum = np.sum(vacuum_dir)

            # If all directions have a vacuum separating the copies, the system
            # represents a finite structure.
            if n_vacuum == 3:

                # Check if system has only one atom
                n_atoms = len(system)
                if n_atoms == 1:
                    atom_comps.append(AtomComponent([0], system))

                # Check if the the system has one or multiple components.
                else:
                    clusters = systax.geometry.get_clusters(system)
                    for cluster in clusters:
                        n_atoms_cluster = len(cluster)
                        if n_atoms_cluster == 1:
                            atom_comps.append(AtomComponent(cluster, system[cluster]))
                        elif n_atoms_cluster > 1:
                            molecule_comps.append(MoleculeComponent(cluster, system[cluster]))

            # 1D structures
            if n_vacuum == 2:

                # Check if the the system has one or multiple components when
                # multiplied once in the periodic dimension
                repetitions = np.invert(vacuum_dir).astype(int)+1
                ext_sys1d = system.repeat(repetitions)
                clusters = systax.geometry.get_clusters(ext_sys1d)
                n_clusters = len(clusters)

                # Find out the dimensions of the system
                is_small = True
                dimensions = systax.geometry.get_dimensions(system, vacuum_dir)
                for i, has_vacuum in enumerate(vacuum_dir):
                    if has_vacuum:
                        dimension = dimensions[i]
                        if dimension > 15:
                            is_small = False

                if n_clusters == 1 and is_small:
                    material1d_comps.append(Material1DComponent(np.arange(len(system)), system.copy()))
                else:
                    unknown_comps.append(UnknownComponent(np.arange(len(system)), system.copy()))

            # 2D structures
            if n_vacuum == 1:

                # Find out whether there is one or more repetitions of a unit
                # cell in the direction orthogonal to the surface. If only one
                # repetition found, the structure is categorized as a 2D
                # material. If more are found, the material is a surface.

                # Find the different clusters
                cluster_indices = systax.geometry.get_clusters(system)

                # Run the surface detection on each cluster (we do not initially know
                # which one is the surface.
                i_surface_comps = []
                for indices in cluster_indices:
                    cluster_system = self.system[indices]
                    i_surf = self._find_surfaces(cluster_system, indices, vacuum_dir)

                    surface_indices = []
                    for surface in i_surf:
                        i_surf_ind = surface.indices
                        surface_indices.extend(i_surf_ind)
                        i_surface_comps.append(surface)

                    indices = set(indices)
                    surface_indices = set(surface_indices)
                    i_misc_ind = indices - surface_indices
                    if i_misc_ind:
                        unknown_comps.append(UnknownComponent(list(i_misc_ind), system[list(i_misc_ind)]))

                # Check that surface components are continuous are chemically
                # connected, and check if they have one or more layers
                for comp in i_surface_comps:

                    # Check if the the system has one or multiple components when
                    # multiplied once in the two periodic dimensions
                    repetitions = np.invert(vacuum_dir).astype(int)+1
                    ext_sys2d = system.repeat(repetitions)
                    clusters = systax.geometry.get_clusters(ext_sys2d)
                    n_clusters = len(clusters)

                    if n_clusters == 1:
                        # Check how many layers are stacked in the direction
                        # orthogonal to the surface
                        if comp.n_layers == 1:
                            material2d_comps.append(Material2DComponent(comp.indices, comp.atoms))
                        elif comp.n_layers > 1:
                            surface_comps.append(comp)
                    else:
                        unknown_comps.append(UnknownComponent(np.arange(len(system)), system.copy()))

                    # Find out the dimensions of the system
                    # is_small = True
                    # dimensions = geometry.get_dimensions(system, vacuum_dir)
                    # dim_2d_threshold = 3 * 2 * max(covalent_radii[system.get_atomic_numbers()])
                    # for i, has_vacuum in enumerate(vacuum_dir):
                        # if has_vacuum:
                            # dimension = dimensions[i]
                            # if dimension > dim_2d_threshold:
                                # is_small = False

            # Bulk structures
            if n_vacuum == 0:

                # Check the number of symmetries
                analyzer = Material3DAnalyzer(system)
                is_crystal = check_if_crystal(analyzer, threshold=self.crystallinity_threshold)

                # Check the number of clusters
                repetitions = [2, 2, 2]
                ext_sys3d = system.repeat(repetitions)
                clusters = systax.geometry.get_clusters(ext_sys3d, self.connectivity_crystal)
                n_clusters = len(clusters)

                if is_crystal and n_clusters == 1:
                    crystal_comps.append(CrystalComponent(
                        np.arange(len(system)),
                        system.copy(),
                        analyzer))
                else:
                    unknown_comps.append(UnknownComponent(np.arange(len(system)), system.copy()))

        # Return a classification for this system.
        n_molecules = len(molecule_comps)
        n_atoms = len(atom_comps)
        n_crystals = len(crystal_comps)
        n_surfaces = len(surface_comps)
        n_material1d = len(material1d_comps)
        n_material2d = len(material2d_comps)
        n_unknown = len(unknown_comps)

        if (n_atoms == 1) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0):
            return Atom(atoms=atom_comps)

        elif (n_atoms == 0) and \
           (n_molecules == 1) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0):
            return Molecule(molecules=molecule_comps)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 1) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0):
            return Material1D(material1d=material1d_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 1) and \
           (n_unknown == 0) and \
           (n_surfaces == 0):
            return Material2D(material2d=material2d_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 1):
            return Surface(surfaces=surface_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 1) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0):
            return Crystal(crystals=crystal_comps)

        else:
            return Unknown(
                atoms=atom_comps,
                molecules=molecule_comps,
                crystals=crystal_comps,
                material1d=material1d_comps,
                material2d=material2d_comps,
                unknowns=unknown_comps,
                surfaces=surface_comps
            )

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

    def _find_surfaces(self, system, orig_indices, vacuum_dir):
        """
        """
        # Find the seed points
        seed_points = self._find_seed_points(system)

        # Find possible bases for each seed point
        surfaces = []
        for seed_index in seed_points:
            possible_spans, neighbour_mask = self._find_possible_bases(system, seed_index)
            best_basis = self._find_best_basis(system, seed_index, possible_spans, neighbour_mask, vacuum_dir)
            print(best_basis)

            n_spans = len(best_basis)
            if best_basis is None or n_spans == 0:
                return []

            # Find the atoms within the found cell
            if n_spans == 3:
                trans_sys = self._find_cell_atoms_3d(system, seed_index, best_basis)
            elif n_spans == 2:
                trans_sys, seed_position = self._find_cell_atoms_2d(system, seed_index, best_basis)

            # Find the adsorbate by looking at translational symmetry. Everything
            # else that does not belong to the surface unit cell is considered an
            # adsorbate.
            surface_indices, n_layers = self._find_surface(system, seed_index, trans_sys)

            # Find the original indices for the surface
            surface_atoms = system[surface_indices]
            orig_surface_indices = []
            for index in list(surface_indices):
                orig_surface_indices.append(orig_indices[index])

            bulk_analyzer = Material3DAnalyzer(trans_sys, spglib_precision=2*self.pos_tol)
            surface = SurfaceComponent(
                orig_surface_indices,
                surface_atoms,
                bulk_analyzer,
                n_layers=n_layers
            )
            surfaces.append(surface)

        return surfaces

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
        indices = systax.geometry.positions_within_basis(positions, cell_basis_vectors, seed_pos, self.pos_tol)
        cell_pos = positions[indices]
        cell_numbers = numbers[indices]

        trans_sys = Atoms(
            cell=cell_basis_vectors,
            positions=cell_pos,
            symbols=cell_numbers
        )

        return trans_sys

    def _find_cell_atoms_2d(self, system, seed_index, cell_basis_vectors):
        """
        Args:
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
        c = 2*self.max_cell_size*c/np.linalg.norm(c)
        test_basis = np.array((a, b, c))
        origin = seed_pos-0.5*c

        # Convert positions to this basis
        indices, rel_cell_pos = systax.geometry.positions_within_basis(
            system,
            test_basis,
            origin,
            self.pos_tol,
            [True, True, False]
        )

        # testi = Atoms(
            # cell=test_basis,
            # scaled_positions=rel_cell_pos,
            # symbols=num[indices]
        # )
        # view(testi)

        # Determine the real cell by getting the maximum and minimum heights of
        # the cell and centering to minimum
        c_comp = rel_cell_pos[:, 2]
        max_index = np.argmax(c_comp)
        min_index = np.argmin(c_comp)
        pos_min = rel_cell_pos[min_index]
        pos_max = rel_cell_pos[max_index]
        new_c = pos_max - pos_min
        new_c_cart = systax.geometry.to_cartesian(test_basis, new_c)
        pos_min_cart = systax.geometry.to_cartesian(test_basis, pos_min)
        cart_cell_pos = systax.geometry.to_cartesian(test_basis, rel_cell_pos)

        # Create a system for the found cell
        new_basis = test_basis
        new_basis[2, :] = new_c_cart
        c_offset = np.array([0, 0, pos_min_cart[2]])
        if np.linalg.norm(new_c_cart) >= 0.1:
            new_pos = systax.geometry.change_basis(
                cart_cell_pos,
                new_basis,
                offset=c_offset)
        else:
            new_pos = cart_cell_pos - c_offset

        new_num = num[indices]
        new_sys = Atoms(
            cell=new_basis,
            positions=new_pos,
            symbols=new_num
        )

        seed_pos = -new_c

        return new_sys, seed_pos

    def _find_orthogonal_direction(self, system, vectors):
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
            repeated = systax.geometry.get_extended_system(system, 15)

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
            # cell = repeated.get_cell()
            dots = np.abs(np.dot(orthogonal_dir, vectors.T))
            orthogonal_vector_index = np.argmax(dots)
            orthogonal_vector = vectors[orthogonal_vector_index]
            orthogonal_dir = orthogonal_vector/np.linalg.norm(orthogonal_vector)

        return orthogonal_dir, orthogonal_vector_index

    def _find_seed_points(self, system):
        """Used to find the given number of seed points where the symmetry
        search is started.

        The search is initiated from the middle of the system, and then
        additional seed points are added the direction orthogonal to the
        surface.
        """
        # from ase.visualize import view
        # view(system)

        orthogonal_dir, _ = self._find_orthogonal_direction(system, system.get_cell())

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

    def _find_surface(self, system, seed_index, surface_unit):
        """Used to find the atoms belonging to a surface.
        """
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

        # searched_coords = set()
        # collection = LinkedUnitCollection(system)
        # self._find_surf_rec(
            # system,
            # collection,
            # number_to_index_map,
            # number_to_pos_map,
            # seed_index,
            # seed_pos,
            # seed_number,
            # surface_unit,
            # searched_coords,
            # (0, 0, 0))

        self._find_surface_recursively(number_to_index_map, number_to_pos_map, indices, cells, 0, 0, 0, system, seed_index, seed_pos, seed_number, surface_unit)

        # Find how many times the unit cell is repeated in the direction
        # orthogonal to the surface
        cell_basis = surface_unit.get_cell()
        ortho_dir, ortho_ind = self._find_orthogonal_direction(system, cell_basis)
        n_layers = 0
        for multiplier in (1, -1):
            done = False
            # print(multiplier)
            i = 0
            start = np.array((0, 0, 0))
            start[ortho_ind] = 1
            while not done:
                i_pos = start*i*multiplier
                i_cell = cells.get(tuple(i_pos))
                # print(i_cell)
                # print(i_pos)
                if i_cell is None:
                    done = True
                else:
                    i += 1
            n_layers += i

        # print(n_layers)

        return indices, n_layers

    def _find_surf_rec(
            self,
            system,
            collection,
            number_to_index_map,
            number_to_pos_map,
            seed_index,
            seed_pos,
            seed_atomic_number,
            unit_cell,
            searched_coords,
            index):

        # Check if this cell has already been searched
        if index in searched_coords:
            return
        else:
            searched_coords.add(index)

        # Transform positions to the new cell basis
        cell_basis = unit_cell.get_cell()
        positions = system.get_positions()
        atomic_numbers = system.get_atomic_numbers()
        pos_shifted = positions - seed_pos
        basis_inverse = np.linalg.inv(cell_basis.T)
        vec_new = np.dot(pos_shifted, basis_inverse.T)

        # For each atom in the basis, find corresponding atom if possible
        cell_pos = unit_cell.get_scaled_positions()
        cell_numbers = unit_cell.get_atomic_numbers()
        new_pos = []
        new_num = []
        new_indices = []
        for i_pos, pos in enumerate(cell_pos):
            disp_tensor = vec_new - pos[np.newaxis, :]
            disp_tensor_cartesian = np.dot(disp_tensor, cell_basis.T)
            dist = np.linalg.norm(disp_tensor_cartesian, axis=1)
            # The tolerance is double here to take into account the possibility
            # that two neighboring cells might be offset from the original cell
            # in opposite directions
            index, = np.where(dist <= self.pos_tol)
            if len(index) != 0:
                new_indices.append(index[0])
                new_pos.append(vec_new[index[0]])
                new_num.append(atomic_numbers[index[0]])
        new_pos = np.array(new_pos)
        new_num = np.array(new_num)

        # If the seed atom was not found for this cell, end the search
        if seed_index is None:
            return

        # Create the new LinkedUnit and add it to the collection representing
        # the surface
        new_unit = LinkedUnit(index, seed_index, seed_pos, cell_basis, new_indices)
        collection.add_unit(new_unit, index)

        # Find the the indices and position of the seed atoms of neighbouring
        # units.
        new_seed_indices = []
        new_seed_pos = []
        multipliers = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        possible_seed_pos = number_to_pos_map[seed_atomic_number]
        possible_seed_indices = number_to_index_map[seed_atomic_number]
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

            a = multiplier[0]
            b = multiplier[1]
            c = multiplier[2]

            new_cell = Atoms(
                cell=i_cell,
                scaled_positions=cell_pos,
                symbols=cell_numbers
            )
            self._find_surf_rec(
                system,
                collection,
                number_to_index_map,
                number_to_pos_map,
                i_seed_index,
                i_seed_pos,
                seed_atomic_number,
                new_cell,
                searched_coords,
                (a, b, c)
            )

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
            unit_cell):
        """A recursive function for traversing the surface and gathering
        indices of the surface atoms.
        """
        # Check if this cell has already been searched
        if (i, j, k) in cells:
            return
        else:
            cells[(i, j, k)] = True

        # Transform positions to the cell basis
        cell_basis = unit_cell.get_cell()
        cell_pos = unit_cell.get_scaled_positions()
        cell_numbers = unit_cell.get_atomic_numbers()
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
        # distance_mask = (seed_span_lengths < 2*self.max_cell_size)
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

            new_cell = Atoms(
                cell=i_cell,
                scaled_positions=cell_pos,
                symbols=cell_numbers
            )
            self._find_surface_recursively(number_to_index_map, number_to_pos_map, indices, cells, a, b, c, system, i_seed_index, i_seed_pos, seed_number, new_cell)

    # def _find_optimal_span(self, syscache, spans):
        # """There might be more than three valid spans that were found, so this
        # function is used to select three.

        # The selection is based on the minimization of the following term:

            # \min e_i^2 + \sum_{i \neq j} \hat{e}_i \cdot \hat{e}_j

        # where the vectors e are the possible unit vectors
        # """
        # n_spans = len(spans)
        # if n_spans > 3:

            # # Get  triplets of spans (combinations)
            # span_indices = range(len(spans))
            # indices = np.array(list(itertools.combinations(span_indices, 3)))

            # # Calculate the 3D array of combined span weights for every triplet
            # norms = syscache["valid_spans_length"]
            # norm1 = norms[indices[:, 0]]
            # norm2 = norms[indices[:, 1]]
            # norm3 = norms[indices[:, 2]]
            # norm_vector = norm1 + norm2 + norm3
            # # print(norm_vector.shape)

            # # Calculate the orthogonality tensor from the dot products
            # dots = syscache["valid_spans_dot"]
            # dot1 = np.abs(dots[indices[:, -1], indices[:, 1]])
            # dot2 = np.abs(dots[indices[:, 1], indices[:, 2]])
            # dot3 = np.abs(dots[indices[:, 0], indices[:, 2]])
            # ortho_vector = dot1 + dot2 + dot3
            # # print(ortho_vector.shape)

            # # Create a combination of the norm tensor and the dots tensor with a
            # # possible weighting
            # norm_weight = 1
            # ortho_weight = 1
            # sum_vector = norm_weight*norm_vector + ortho_weight*ortho_vector

            # # Sort the triplets by value
            # i = indices[:, 0]
            # j = indices[:, 1]
            # k = indices[:, 2]
            # idx = sum_vector.argsort()
            # indices = np.dstack((i[idx], j[idx], k[idx]))

            # # Use the span with lowest score
            # cell = spans[indices[0, 0, :]]
        # else:
            # cell = spans

        # # Choose linearly independent vectors
        # n_spans = len(cell)
        # if n_spans == 3:
            # lens = np.linalg.norm(cell, axis=1)
            # norm_cell = cell/lens
            # vol = np.linalg.det(norm_cell)
            # if vol < 0.1:
                # cell = cell[0:1, :]

        # return cell
