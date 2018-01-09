from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import ase.build
from ase.visualize import view

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
from systax.analysis.class3danalyzer import Class3DAnalyzer
from systax.symmetry import check_if_crystal
from systax.classification.periodicfinder import PeriodicFinder

__metaclass__ = type


class SystemCache(dict):
    pass


class Classifier():
    """A class that is used to analyze the contents of an atomistic system and
    separate the consituent atoms into different components along with some
    meaningful additional information.
    """
    def __init__(
            self,
            seed_position="cm",
            max_cell_size=8,
            pos_tol=0.3,
            pos_tol_mode="relative",
            angle_tol=20,
            cluster_threshold=3.0,
            crystallinity_threshold=0.25,
            delaunay_threshold=1.5,
            delaunay_threshold_mode="relative",
            pos_tol_factor=2,
            n_edge_tol=0.95,
            cell_size_tol=0.25
            ):
        """
        Args:
            seed_position(str or np.ndarray): The seed position. Either provide
                a 3D vector from which the closest atom will be used as a seed, or
                then provide a valid option as a string. Valid options are:
                    - 'cm': One seed nearest to center of mass
            max_cell_size(float): The maximum cell size
            pos_tol(float): The position tolerance in angstroms for finding translationally
                repeated units.
            pos_tol_mode(str): The mode for calculting the position tolerance.
                One of the following:
                    - "relative": Tolerance relative to the average nearest
                      neighbour distance.
                    - "absolute": Absolute tolerance in angstroms.
            cluster_threshold(float): A parameter that controls which atoms are
                considered to be energetically connected when clustering is
                perfomed the connectivity that is required for .
            crystallinity_threshold(float): The threshold of number of symmetry
                operations per atoms in primitive cell that is required for
                crystals.
        """
        self.max_cell_size = max_cell_size
        self.pos_tol = pos_tol
        self.abs_pos_tol = None
        self.pos_tol_mode = pos_tol_mode
        self.angle_tol = angle_tol
        self.crystallinity_threshold = crystallinity_threshold
        self.cluster_threshold = cluster_threshold
        self.delaunay_threshold = delaunay_threshold
        self.abs_delaunay_threshold = None
        self.delaunay_threshold_mode = delaunay_threshold_mode
        self.pos_tol_factor = pos_tol_factor
        self.n_edge_tol = n_edge_tol
        self.cell_size_tol = cell_size_tol

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

    def classify(self, system):
        """A function that analyzes the system and breaks it into different
        components.

        Args:
            system(ASE.Atoms or System): Atomic system to classify.

        Returns:
            Classification: One of the subclasses of the Classification base
            class that represents a classification.
        """
        self.system = system
        classification = None

        # Calculate the displacement tensor for the original system. It will be
        # reused in multiple sections.
        pos = system.get_positions()
        cell = system.get_cell()
        pbc = system.get_pbc()
        disp_tensor = systax.geometry.get_displacement_tensor(pos, pos)
        if pbc.any():
            disp_tensor_pbc = systax.geometry.get_displacement_tensor(pos, pos, cell, pbc, mic=True)
        else:
            disp_tensor_pbc = disp_tensor
        dist_matrix_pbc = np.linalg.norm(disp_tensor, axis=2)

        # If pos_tol_mode or delaunay_threshold_mode is relative, get the
        # average distance to closest neighbours
        if self.pos_tol_mode == "relative" or self.delaunay_threshold_mode == "relative":
            min_basis = np.linalg.norm(cell, axis=1).min()
            dist_matrix_mod = np.array(dist_matrix_pbc)
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
            dimensionality, vacuum_dir = systax.geometry.get_dimensionality(
                system,
                self.cluster_threshold,
                disp_tensor=disp_tensor,
                disp_tensor_pbc=disp_tensor_pbc
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

            classification = Class1D(vacuum_dir)

            # Find out the dimensions of the system
            is_small = True
            dimensions = systax.geometry.get_dimensions(system, vacuum_dir)
            for i, has_vacuum in enumerate(vacuum_dir):
                if has_vacuum:
                    dimension = dimensions[i]
                    if dimension > 15:
                        is_small = False

            if is_small:
                classification = Material1D(vacuum_dir)

        # 2D structures
        elif dimensionality == 2:

            classification = Class2D(vacuum_dir)

            # Get the index of the seed atom
            if self.seed_position == "cm":
                seed_vec = self.system.get_center_of_mass()
            else:
                seed_vec = self.seed_position
            seed_index = systax.geometry.get_nearest_atom(self.system, seed_vec)

            # Run the region detection on the whole system.
            periodicfinder = PeriodicFinder(
                pos_tol=self.abs_pos_tol,
                angle_tol=self.angle_tol,
                max_cell_size=self.max_cell_size,
                pos_tol_factor=self.pos_tol_factor,
                cell_size_tol=self.cell_size_tol,
                n_edge_tol=self.n_edge_tol
            )
            region = periodicfinder.get_region(system, seed_index, disp_tensor_pbc, vacuum_dir, self.abs_delaunay_threshold)
            if region is not None:
                region = region[1]

                # If the region covers less than 50% of the whole system,
                # categorize as Class2D
                n_region_atoms = len(region.get_basis_indices())
                n_atoms = len(system)
                coverage = n_region_atoms/n_atoms

                if coverage >= 0.5:
                    if region.is_2d:
                        classification = Material2D(vacuum_dir, region)
                    else:
                        classification = Surface(vacuum_dir, region)

        # Bulk structures
        elif dimensionality == 3:

            classification = Class3D(vacuum_dir)

            # Check the number of symmetries
            analyzer = Class3DAnalyzer(system)
            is_crystal = check_if_crystal(analyzer, threshold=self.crystallinity_threshold)

            # If the structure is connected but the symmetry criteria was
            # not fullfilled, check the number of atoms in the primitive
            # cell. If above a certain threshold, try to find periodic
            # region to see if it is a crystal containing a defect.
            if not is_crystal:
                primitive_system = analyzer.get_primitive_system()
                n_atoms_prim = len(primitive_system)
                if n_atoms_prim >= 20:
                    periodicfinder = PeriodicFinder(
                        pos_tol=self.abs_pos_tol,
                        angle_tol=self.angle_tol,
                        max_cell_size=self.max_cell_size,
                        pos_tol_factor=self.pos_tol_factor,
                        cell_size_tol=self.cell_size_tol,
                        n_edge_tol=self.n_edge_tol
                    )

                    # Get the index of the seed atom
                    if self.seed_position == "cm":
                        seed_vec = self.system.get_center_of_mass()
                    else:
                        seed_vec = self.seed_position
                    seed_index = systax.geometry.get_nearest_atom(self.system, seed_vec)

                    region = periodicfinder.get_region(system, seed_index, disp_tensor_pbc, vacuum_dir, self.abs_delaunay_threshold)
                    if region is not None:
                        region = region[1]

                        # If all the regions cover at least 80% of the structure,
                        # then we consider it to be a defected crystal
                        n_region_atoms = len(region.get_basis_indices())
                        n_atoms = len(system)
                        coverage = n_region_atoms/n_atoms
                        if coverage >= 0.5:
                            classification = Crystal(analyzer, region=region)

            elif is_crystal:
                classification = Crystal(analyzer)

        return classification
