import numpy as np

# from ase.visualize import view
from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii

from matid.classifications import \
    Surface, \
    Atom, \
    Material2D, \
    Unknown, \
    Class0D, \
    Class1D, \
    Class2D, \
    Class3D
    # Crystal
import matid.geometry
from matid.data import constants
from matid.classification.periodicfinder import PeriodicFinder
# from matid.symmetry.symmetryanalyzer import SymmetryAnalyzer


class Classifier():
    """A class that is used to analyze the contents of an atomistic system and
    separate the consituent atoms into different components along with other
    meaningful information.
    """
    def __init__(
            self,
            seed_position="cm",
            max_cell_size=constants.MAX_CELL_SIZE,
            pos_tol=None,
            pos_tol_mode="relative",
            pos_tol_scaling=constants.POS_TOL_SCALING,
            angle_tol=constants.ANGLE_TOL,
            cluster_threshold=constants.CLUSTER_THRESHOLD,
            radii="covalent",
            crystallinity_threshold=constants.CRYSTALLINITY_THRESHOLD,
            delaunay_threshold=constants.DELAUNAY_THRESHOLD,
            bond_threshold=constants.BOND_THRESHOLD,
            delaunay_threshold_mode="relative",
            chem_similarity_threshold=constants.CHEM_SIMILARITY_THRESHOLD,
            cell_size_tol=constants.CELL_SIZE_TOL,
            max_n_atoms=constants.MAX_N_ATOMS,
            max_2d_cell_height=constants.MAX_2D_CELL_HEIGHT,
            max_2d_single_cell_size=constants.MAX_SINGLE_CELL_SIZE,
            symmetry_tol=constants.SYMMETRY_TOL,
            min_coverage=constants.MIN_COVERAGE
            ):
        """
        Args:
            seed_position(str | np.ndarray): The seed position. Either provide
                a 3D vector from which the closest atom will be used as a seed, or
                then provide a valid option as a string. Valid options are:
                    - 'cm': One seed nearest to center of mass
            max_cell_size(float): The maximum size of cell basis vectors.
            pos_tol(float): The position tolerance for finding translationally
                repeated units. The units depend on 'pos_tol_mode'.
            pos_tol_mode(str): The mode for calculating the position tolerance.
                One of the following:
                    - "relative": Tolerance relative to the average nearest
                      neighbour distance.
                    - "absolute": Absolute tolerance in angstroms.
            pos_tol_scaling(float): The distance dependent scaling factor for
                the positions tolerance.
            angle_tol(float): The angle below which vectors in the cell basis are
                considered to be parallel.
            cluster_threshold(float): A parameter that controls which atoms are
                considered to be energetically connected when clustering is
                perfomed. Given in angstroms.
            radii(str | np.ndarray): The radii to use for atoms. Use either a preset
                or a custom list or atomic radii where the atomic number is used as an
                index. The available presets are:

                    - covalent: Covalent radii from DOI:10.1039/B801115J
                    - vdw: van Der Waals radii from DOI:10.1039/C3DT50599E
                    - vdw_covalent: preferably van Der Waals radii, covalent if vdw not defined.

            crystallinity_threshold(float): The threshold of number of symmetry
                operations per atoms in primitive cell that is required for
                crystals.
            max_n_atoms(int): The maximum number of atoms in the system. If the
                system has more atoms than this, a ValueError is raised. If
                undefined, there is no maximum.
            bond_threshold(float): The clustering threshold when determining
                the connectivity of atoms in a surface or 2D-material.
            delaunay_threshold(str): The maximum length of an edge in the
                Delaunay triangulation.
            delaunay_threshold_mode(str): The mode for calculating the maximum
                length of an edge in the Delaunay triangulation.
                One of the following:
                    - "relative": Tolerance relative to the average nearest
                      neighbour distance.
                    - "absolute": Absolute tolerance in angstroms.
            pos_tol_factor(float): The factor for multiplying the position
                tolerance when searching neighbouring cell seed atoms.
            cell_size_tol(float): The tolerance for cell sizes to be considered
                equal. Given relative to the smallest cell size.
            max_2d_cell_height(float): The maximum allowed thickness in for a 2D
                material. Given in angstroms.
            max_2d_single_cell_size(float): The maximum allowed cell size for
                2D materials with only one unit cell found in the simulation
                cell. Given in angstroms.
            symmetry_tol(float): The tolerance for finding symmetry positions
                when determining the conventional cell for bulk structures. Given
                in angstroms.
            min_coverage(float): The minimum fraction that a found periodic
                region has to cover in the structure for the entire structure to be
                classified based on that found region (as surface or 2D
                material).
        """
        if pos_tol_mode == "relative" and pos_tol is None:
            pos_tol = constants.REL_POS_TOL
        if isinstance(max_cell_size, (int, float)):
            max_cell_size = [max_cell_size]
        self.max_cell_size = max_cell_size
        if isinstance(pos_tol, (int, float)):
            pos_tol = [pos_tol]
        self.pos_tol = pos_tol
        self.pos_tol_scaling = pos_tol_scaling
        self.abs_pos_tol = None
        self.pos_tol_mode = pos_tol_mode
        self.angle_tol = angle_tol
        self.crystallinity_threshold = crystallinity_threshold
        self.cluster_threshold = cluster_threshold
        self.radii = radii
        self.delaunay_threshold = delaunay_threshold
        self.abs_delaunay_threshold = None
        self.delaunay_threshold_mode = delaunay_threshold_mode
        self.bond_threshold = bond_threshold
        self.chem_similarity_threshold = chem_similarity_threshold
        self.pos_tol_scaling = pos_tol_scaling
        self.cell_size_tol = cell_size_tol
        self.max_n_atoms = max_n_atoms
        self.max_2d_cell_height = max_2d_cell_height
        self.max_2d_single_cell_size = max_2d_single_cell_size
        self.symmetry_tol = symmetry_tol
        self.min_coverage = min_coverage

        # Check seed position
        if type(seed_position) == str:
            if seed_position == "cm":
                pass
            else:
                raise ValueError(
                    "Unknown seed_position: '{}'. Please provide a 3D vector "
                    "or a valid option as a string.".format(seed_position)
                )
        self.seed_position = seed_position

        # Check pos tolerance mode
        allowed_modes = set(["relative", "absolute"])
        if pos_tol_mode not in allowed_modes:
            raise ValueError("Unknown value '{}' for 'pos_tol_mode'.".format(pos_tol_mode))

        # Check delaunay tolerance mode
        allowed_modes = set(["relative", "absolute"])
        if delaunay_threshold_mode not in allowed_modes:
            raise ValueError("Unknown value '{}' for 'delaunay_threshold_mode'.".format(delaunay_threshold_mode))

    def classify(self, input_system):
        """A function that analyzes the system and breaks it into different
        components.

        Args:
            system(ASE.Atoms or System): Atomic system to classify.

        Returns:
            Classification: One of the subclasses of the Classification base
            class that represents a classification.

        Raises:
            ValueError: If the system has more atoms than self.max_n_atoms
        """
        # We wrap the positions to to be inside the cell.
        system = input_system.copy()
        system.wrap()
        self.system = system
        classification = None

        n_atoms = len(system)
        if n_atoms > self.max_n_atoms:
            raise ValueError(
                "The system contains more atoms ({}) than the current allowed "
                "limit of {}. If you wish you can increase this limit with the "
                "max_n_atoms attribute.".format(n_atoms, self.max_n_atoms)
            )

        # Calculate the displacement tensor for the original system. It will be
        # reused in multiple sections.
        cell = system.get_cell()
        distances = matid.geometry.get_distances(system)

        # If pos_tol_mode or delaunay_threshold_mode is relative, get the
        # average distance to closest neighbours
        num = system.get_atomic_numbers()
        if self.pos_tol_mode == "relative" or self.delaunay_threshold_mode == "relative":
            min_basis = np.linalg.norm(cell, axis=1).min()
            dist_matrix_mod = np.array(distances.dist_matrix_mic)
            np.fill_diagonal(dist_matrix_mod, min_basis)
            global_min_dist = dist_matrix_mod.min()
            min_dist = np.min(dist_matrix_mod, axis=1)
            mean_min_dist = min_dist.mean()

            if self.pos_tol_mode == "relative":
                self.abs_pos_tol = np.array(self.pos_tol)*global_min_dist
            elif self.pos_tol_mode == "absolute":
                self.abs_pos_tol = self.pos_tol

            if self.delaunay_threshold_mode == "relative":
                self.abs_delaunay_threshold = self.delaunay_threshold * mean_min_dist
            elif self.delaunay_threshold_mode == "absolute":
                self.abs_delaunay_threshold = self.delaunay_threshold

        # Get the system dimensionality
        dimensionality = matid.geometry.get_dimensionality(
            system,
            self.cluster_threshold,
            distances.dist_matrix_radii_mic
        )
        if dimensionality is None:
            return Unknown(input_system)

        # 0D structures
        if dimensionality == 0:
            classification = Class0D(input_system)

            # Systems with one atom have their own classification.
            n_atoms = len(system)
            if n_atoms == 1:
                classification = Atom(input_system)

        # 1D structures
        elif dimensionality == 1:
            classification = Class1D(input_system)

        # 2D structures
        elif dimensionality == 2:

            classification = Class2D(input_system)

            # Get the indices of the used seed atoms
            seed_indices = []
            test_sys = system.copy()
            cm = matid.geometry.get_center_of_mass(test_sys)

            # If center of mass defined, for each atomic element find the
            # occurrence closest to center of mass to use as seed point.
            num = self.system.get_atomic_numbers()
            elems = set(num)

            if self.seed_position == "cm":
                dist = np.linalg.norm(system.get_positions() - cm, axis=1)
                indices = np.argsort(dist)
                for i in indices:
                    i_elem = num[i]
                    if i_elem in elems:
                        seed_indices.append(i)
                        elems.remove(i_elem)
                    if len(elems) == 0:
                        break
            else:
                if type(self.seed_position) == int:
                    seed_indices = [self.seed_position]
                elif isinstance(self.seed_position, (tuple, list, np.ndarray)):
                    seed_indices = self.seed_position

            # Find the best region by trying out different parameters options
            best_region = self.cross_validate_region(
                system,
                seed_indices,
                distances
            )

            if best_region is not None:
                # Check that the region was connected cyclically in two
                # directions. This ensures that finite systems or systems
                # with a dislocation at the cell boundary are filtered.
                region_conn = best_region.get_connected_directions()
                n_region_conn = np.sum(region_conn)
                region_is_periodic = n_region_conn == 2

                # This might be unnecessary because the connectivity of the
                # unit cell is already checked.
                clusters = best_region.get_clusters()
                basis_indices = set(list(best_region.get_basis_indices()))
                split = True
                for cluster in clusters:
                    if basis_indices.issubset(cluster):
                        split = False

                # Check that the found region covers enough of the entire
                # system. If not, then the region alone cannot be used to
                # classify the entire structure. This happens e.g. when one
                # 2D sheet is found from a 2D heterostructure, or a local
                # pattern is found inside a structure.
                n_atoms = len(system)
                n_basis_atoms = len(basis_indices)
                coverage = n_basis_atoms/n_atoms
                covered = coverage >= self.min_coverage

                if not split and covered and region_is_periodic:
                    if best_region.is_2d:
                        classification = Material2D(input_system, best_region)
                    else:
                        classification = Surface(input_system, best_region)

        # Bulk structures
        elif dimensionality == 3:
            classification = Class3D(input_system)

        return classification

    def cross_validate_region(
            self,
            system,
            seed_indices,
            distances,
        ):
        """Given a system tries multiple combinations of different search
        parameters to find a prototype cell and a corresponding region that
        best explains the underlying structure.

        Args:
        Returns:
        """
        # Run the detection with multiple position tolerances
        best_region = None
        most_atoms = 0

        # Here a cross-validation is performed to choose parameters that
        # produce best results. The performance of the parameters is
        # quantified by counting the number of valid atoms in the region.
        # The search is stopped if a system with zero outliers is found.
        n_atoms = len(system)
        for index in seed_indices:
            for size in self.max_cell_size:
                for tol in self.abs_pos_tol:

                    # Run the region detection on the whole system.
                    periodicfinder = PeriodicFinder(
                        angle_tol=self.angle_tol,
                        pos_tol_scaling=self.pos_tol_scaling,
                        cell_size_tol=self.cell_size_tol,
                        max_2d_cell_height=self.max_2d_cell_height,
                        max_2d_single_cell_size=self.max_2d_single_cell_size,
                        chem_similarity_threshold=self.chem_similarity_threshold
                    )
                    region = periodicfinder.get_region(
                        system,
                        index,
                        size,
                        tol,
                        self.abs_delaunay_threshold,
                        self.bond_threshold,
                        distances
                    )

                    if region is not None:
                        basis_indices = region.get_basis_indices()
                        n_basis = len(basis_indices)

                        # There are no outliers with this cell, other
                        # options do not need to explored.
                        if n_basis == n_atoms:
                            return region

                        # Store this region if it is better than the
                        # previous best
                        if n_basis > most_atoms:
                            most_atoms = n_basis
                            best_region = region
        return best_region
