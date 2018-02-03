from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import ase.build
from ase.visualize import view
from ase.data import covalent_radii

from systax.exceptions import SystaxError

from systax.classification.classifications import \
    Surface, \
    Atom, \
    Molecule, \
    Crystal, \
    Material1D, \
    Material2D, \
    Unknown, \
    Class0D, \
    Class1D, \
    Class2D, \
    Class3D
import systax.geometry
from systax.data import constants
from systax.analysis.class3danalyzer import Class3DAnalyzer
from systax.analysis.class2danalyzer import Class2DAnalyzer
from systax.symmetry import check_if_crystal
from systax.classification.periodicfinder import PeriodicFinder

__metaclass__ = type


class Classifier():
    """A class that is used to analyze the contents of an atomistic system and
    separate the consituent atoms into different components along with some
    meaningful additional information.
    """
    def __init__(
            self,
            seed_position="cm",
            max_cell_size=None,
            pos_tol=None,
            pos_tol_mode="relative",
            pos_tol_scaling=None,
            angle_tol=None,
            cluster_threshold=None,
            crystallinity_threshold=None,
            delaunay_threshold=None,
            bond_threshold=None,
            delaunay_threshold_mode="relative",
            chem_env_threshold=None,
            n_edge_tol=None,
            cell_size_tol=None,
            max_n_atoms=None,
            coverage_threshold=None,
            max_vacancy_ratio=None,
            max_2d_cell_height=None
            ):
        """
        Args:
            seed_position(str or np.ndarray): The seed position. Either provide
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
                perfomed.
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
            n_edge_tol(float): The minimum fraction of edges that have to be in the
                periodicity graph for the cell to be considered valid. Given
                relative to the cell with maximum number of edges.
            cell_size_tol(float): The tolerance for cell sizes to be considered
                equal. Given relative to the smallest cell size.
            coverage_threshold(float): The fraction of atoms that have to
                belong to a region in order for the system to be considered surface
                or 2D-material.
            max_vacancy_ratio(float): Maximum fraction of vacancy atoms/atoms
                in the region for the system to be considered valid. If too
                many vacancies are found, the classification will be left more
                generic, e.g. Class2D.
        """
        if max_cell_size is None:
            max_cell_size = constants.MAX_CELL_SIZE
        if pos_tol_mode == "relative" and pos_tol is None:
            pos_tol = constants.REL_POS_TOL
        if pos_tol_scaling is None:
            pos_tol_scaling = constants.POS_TOL_SCALING
        if angle_tol is None:
            angle_tol = constants.ANGLE_TOL
        if crystallinity_threshold is None:
            crystallinity_threshold = constants.CRYSTALLINITY_THRESHOLD
        if cluster_threshold is None:
            cluster_threshold = constants.CLUSTER_THRESHOLD
        if delaunay_threshold is None:
            delaunay_threshold = constants.DELAUNAY_THRESHOLD
        if bond_threshold is None:
            bond_threshold = constants.BOND_THRESHOLD
        if chem_env_threshold is None:
            chem_env_threshold = constants.CHEM_ENV_THRESHOLD
        if n_edge_tol is None:
            n_edge_tol = constants.N_EDGE_TOL
        if cell_size_tol is None:
            cell_size_tol = constants.CELL_SIZE_TOL
        if max_n_atoms is None:
            max_n_atoms = constants.MAX_N_ATOMS
        if coverage_threshold is None:
            coverage_threshold = constants.COVERAGE_THRESHOLD
        if max_vacancy_ratio is None:
            max_vacancy_ratio = constants.MAX_VACANCY_RATIO
        if max_2d_cell_height is None:
            max_2d_cell_height = constants.MAX_2D_CELL_HEIGHT

        self.max_cell_size = max_cell_size
        self.pos_tol = pos_tol
        self.pos_tol_scaling = pos_tol_scaling
        self.abs_pos_tol = None
        self.pos_tol_mode = pos_tol_mode
        self.angle_tol = angle_tol
        self.crystallinity_threshold = crystallinity_threshold
        self.cluster_threshold = cluster_threshold
        self.delaunay_threshold = delaunay_threshold
        self.abs_delaunay_threshold = None
        self.delaunay_threshold_mode = delaunay_threshold_mode
        self.bond_threshold = bond_threshold
        self.chem_env_threshold = chem_env_threshold
        self.pos_tol_scaling = pos_tol_scaling
        self.n_edge_tol = n_edge_tol
        self.cell_size_tol = cell_size_tol
        self.max_n_atoms = max_n_atoms
        self.coverage_threshold = coverage_threshold
        self.max_vacancy_ratio = max_vacancy_ratio
        self.max_2d_cell_height = max_2d_cell_height

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

        # Check delunay tolerance mode
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
        pos = system.get_positions()
        cell = system.get_cell()
        pbc = system.get_pbc()
        disp_tensor = systax.geometry.get_displacement_tensor(pos, pos)
        if pbc.any():
            disp_tensor_pbc, disp_factors = systax.geometry.get_displacement_tensor(
                pos,
                pos,
                cell,
                pbc,
                mic=True,
                return_factors=True
            )
        else:
            disp_tensor_pbc = disp_tensor
            disp_factors = np.zeros(disp_tensor.shape)
        dist_matrix_pbc = np.linalg.norm(disp_tensor_pbc, axis=2)

        # Calculate the distance matrix where the periodicity and the covalent
        # radii have been taken into account
        dist_matrix_radii_pbc = np.array(dist_matrix_pbc)
        num = system.get_atomic_numbers()
        radii = covalent_radii[num]
        radii_matrix = radii[:, None] + radii[None, :]
        dist_matrix_radii_pbc -= radii_matrix

        # If pos_tol_mode or delaunay_threshold_mode is relative, get the
        # average distance to closest neighbours
        if self.pos_tol_mode == "relative" or self.delaunay_threshold_mode == "relative":
            min_basis = np.linalg.norm(cell, axis=1).min()
            dist_matrix_mod = np.array(dist_matrix_pbc)
            dist_matrix_mod /= radii_matrix
            np.fill_diagonal(dist_matrix_mod, min_basis)
            min_dist = np.min(dist_matrix_mod, axis=1)
            mean_min_dist = min_dist.mean()

            if self.pos_tol_mode == "relative":
                self.abs_pos_tol = self.pos_tol * mean_min_dist
            elif self.pos_tol_mode == "absolute":
                self.abs_pos_tol = self.pos_tol

            if self.delaunay_threshold_mode == "relative":
                self.abs_delaunay_threshold = self.delaunay_threshold * mean_min_dist
            elif self.delaunay_threshold_mode == "absolute":
                self.abs_delaunay_threshold = self.delaunay_threshold

        # Get the system dimensionality
        try:
            dimensionality = systax.geometry.get_dimensionality(
                system,
                self.cluster_threshold,
                dist_matrix_radii_pbc
            )
        except SystaxError:
            return Unknown()

        # 0D structures
        if dimensionality == 0:
            classification = Class0D()

            # Check if system consists of one atom or one molecule
            n_atoms = len(system)
            if n_atoms == 1:
                classification = Atom()
            else:
                formula = system.get_chemical_formula()
                try:
                    ase.build.molecule(formula)
                except KeyError:
                    pass
                else:
                    classification = Molecule()

        # 1D structures
        elif dimensionality == 1:

            classification = Class1D()

            # Find out the dimensions of the system
            # is_small = True
            # dimensions = systax.geometry.get_dimensions(system)
            # for i, has_vacuum in enumerate(vacuum_dir):
                # if has_vacuum:
                    # dimension = dimensions[i]
                    # if dimension > 15:
                        # is_small = False

            # if is_small:
                # classification = Material1D(vacuum_dir)

        # 2D structures
        elif dimensionality == 2:

            classification = Class2D()

            # Get the index of the seed atom
            if self.seed_position == "cm":
                # seed_vec = system.get_center_of_mass()
                seed_vec = systax.geometry.get_center_of_mass(system)
            else:
                seed_vec = self.seed_position

            seed_index = systax.geometry.get_nearest_atom(self.system, seed_vec)

            # Run the region detection on the whole system.
            periodicfinder = PeriodicFinder(
                angle_tol=self.angle_tol,
                max_cell_size=self.max_cell_size,
                # pos_tol_factor=self.pos_tol_factor,
                pos_tol_scaling=self.pos_tol_scaling,
                cell_size_tol=self.cell_size_tol,
                n_edge_tol=self.n_edge_tol,
                max_2d_cell_height=self.max_2d_cell_height,
                chem_env_threshold=self.chem_env_threshold
            )
            region = periodicfinder.get_region(
                system,
                seed_index,
                self.abs_pos_tol,
                self.abs_delaunay_threshold,
                self.bond_threshold,
                disp_tensor_pbc,
                disp_factors,
                disp_tensor,
                dist_matrix_radii_pbc,
            )
            if region is not None:
                region = region[1]

                # If the basis atoms in the region are split into multiple
                # disconnected pieces (as indicated by clustering), then they
                # cannot be classified
                clusters = region.get_clusters()
                basis_indices = set(list(region.get_basis_indices()))
                split = True
                for cluster in clusters:
                    if basis_indices.issubset(cluster):
                        split = False

                if not split:
                    # If the region covers less than 50% of the whole system,
                    # categorize as Class2D
                    n_region_atoms = len(region.get_basis_indices())
                    n_atoms = len(system)
                    coverage = n_region_atoms/n_atoms
                    n_vacancies = len(region.get_vacancies())
                    vacancy_ratio = n_vacancies/n_region_atoms

                    if coverage >= self.coverage_threshold and vacancy_ratio <= self.max_vacancy_ratio:
                        if region.is_2d:
                            # The Class2DAnalyzer needs to know which direcion
                            # in the cell is not periodic. Now that the cell
                            # has been found, we know that the third axis is
                            # set as the non-periodic one.
                            analyzer = Class2DAnalyzer(region.cell, vacuum_gaps=[False, False, True])
                            classification = Material2D(region, analyzer)
                        else:
                            analyzer = Class3DAnalyzer(region.cell)
                            classification = Surface(region, analyzer)

        # Bulk structures
        elif dimensionality == 3:

            classification = Class3D()

            # Check the number of symmetries
            analyzer = Class3DAnalyzer(system)
            is_crystal = check_if_crystal(analyzer, threshold=self.crystallinity_threshold)

            # If the structure is connected but the symmetry criteria was
            # not fullfilled, check the number of atoms in the primitive
            # cell. If above a certain threshold, try to find periodic
            # region to see if it is a crystal containing a defect.
            if not is_crystal:
                pass

                # This section is currently disabled. Can be reenabled once
                # more extensive testing is carried out on the detection of
                # defects in crystals.

                # primitive_system = analyzer.get_primitive_system()
                # n_atoms_prim = len(primitive_system)
                # if n_atoms_prim >= 20:
                    # periodicfinder = PeriodicFinder(
                        # pos_tol=self.abs_pos_tol,
                        # angle_tol=self.angle_tol,
                        # max_cell_size=self.max_cell_size,
                        # pos_tol_factor=self.pos_tol_factor,
                        # cell_size_tol=self.cell_size_tol,
                        # n_edge_tol=self.n_edge_tol
                    # )

                    # # Get the index of the seed atom
                    # if self.seed_position == "cm":
                        # seed_vec = self.system.get_center_of_mass()
                    # else:
                        # seed_vec = self.seed_position
                    # seed_index = systax.geometry.get_nearest_atom(self.system, seed_vec)

                    # region = periodicfinder.get_region(system, seed_index, disp_tensor_pbc, disp_tensor, self.abs_delaunay_threshold)
                    # if region is not None:
                        # region = region[1]

                        # # If all the regions cover at least 80% of the structure,
                        # # then we consider it to be a defected crystal
                        # n_region_atoms = len(region.get_basis_indices())
                        # n_atoms = len(system)
                        # coverage = n_region_atoms/n_atoms
                        # if coverage >= self.coverage_threshold:
                            # classification = Crystal(analyzer, region=region)

            elif is_crystal:
                classification = Crystal(analyzer)

        return classification
