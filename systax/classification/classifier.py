from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


import itertools

import numpy as np

from ase import Atoms
from ase.data import covalent_radii
from ase.visualize import view

from systax.exceptions import ClassificationError
from systax.classification.components import SurfaceComponent, SurfacePristineComponent, AtomComponent, MoleculeComponent, CrystalComponent, Material1DComponent, Material2DComponent, Material2DPristineComponent, UnknownComponent
from systax.classification.classifications import Surface, SurfacePristine, Atom, Molecule, Crystal, Material1D, Material2D, Material2DPristine, Unknown
import systax.geometry
from systax.analysis.class3danalyzer import Class3DAnalyzer
from systax.analysis.class2danalyzer import Class2DAnalyzer
from systax.core.linkedunits import LinkedUnitCollection, LinkedUnit
from systax.symmetry import check_if_crystal
from systax.core.system import System
from systax.classification.periodicfinder import PeriodicFinder


class SystemCache(dict):
    pass


class Classifier():
    """A class that is used to analyze the contents of an atomistic system and
    separate the consituent atoms into different components along with some
    meaningful additional information.
    """
    def __init__(
            self,
            seed_algorithm="cm",
            max_cell_size=3,
            pos_tol=0.5,
            n_seeds=2,
            crystallinity_threshold=0.1,
            connectivity_crystal=1.9,
            thickness_2d=6,
            layers_2d=1
            ):
        """
        Args:
            seed_algorithm(str): Algorithm for finding unit cells. The options are:
                -"cm": One seed point at atom nearest to center of mass.
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
        self.seed_algorithm = seed_algorithm
        self.max_cell_size = max_cell_size
        self.pos_tol = pos_tol
        self.n_seeds = 1
        self.crystallinity_threshold = crystallinity_threshold
        self.connectivity_crystal = connectivity_crystal
        self.thickness_2d = thickness_2d
        self.layers_2d = layers_2d
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
        surface_prist_comps = []
        atom_comps = []
        molecule_comps = []
        crystal_comps = []
        unknown_comps = []
        material1d_comps = []
        material2d_comps = []
        material2d_prist_comps = []

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

                # Run the surface detection on the whole system.
                total_regions = []
                indices = list(range(len(system)))
                periodicfinder = PeriodicFinder(
                    pos_tol=self.pos_tol,
                    seed_algorithm="cm",
                    max_cell_size=self.max_cell_size)
                regions = periodicfinder.get_regions(system, vacuum_dir)

                surface_indices = set()
                for i_region in regions:

                    i_indices, i_unit_collection, i_region_atoms, i_cell = i_region
                    if len(i_indices) != 0:
                        total_regions.append(i_region)
                        surface_indices.update(i_indices)

                indices = set(indices)
                i_misc_ind = indices - surface_indices
                if i_misc_ind:
                    unknown_comps.append(UnknownComponent(list(i_misc_ind), system[list(i_misc_ind)]))

                # Check that surface components are continuous are chemically
                # connected, and check if they have one or more layers
                for i_region in total_regions:

                    i_orig_indices, i_unit_collection, i_atoms, i_cell = i_region

                    # Check if the the system has one or multiple components when
                    # multiplied once in the two periodic dimensions
                    repetitions = np.invert(vacuum_dir).astype(int)+1
                    ext_sys2d = i_atoms.repeat(repetitions)
                    clusters = systax.geometry.get_clusters(ext_sys2d)
                    n_clusters = len(clusters)

                    if n_clusters == 1:
                        # Check the number of layers, the thickness and the
                        # pristinity
                        analyzer_2d = Class2DAnalyzer(i_atoms, vacuum_gaps=vacuum_dir, unitcollection=i_unit_collection, unit_cell=i_cell)
                        is_pristine = analyzer_2d.is_pristine()
                        layer_mean, layer_std = analyzer_2d.get_layer_statistics()
                        thickness = analyzer_2d.get_thickness()

                        if layer_mean == self.layers_2d and layer_std == 0.0 and thickness < self.thickness_2d:
                            if is_pristine:
                                material2d_prist_comps.append(Material2DPristineComponent(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                            # else:
                                # material2d_comps.append(Material2DComponent(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                        elif layer_mean > self.layers_2d:
                            if is_pristine:
                                surface_prist_comps.append(SurfacePristineComponent(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                            # else:
                                # surface_comps.append(SurfaceComponent(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                    else:
                        unknown_comps.append(UnknownComponent(np.arange(len(system)), system.copy()))

            # Bulk structures
            if n_vacuum == 0:

                # Check the number of symmetries
                analyzer = Class3DAnalyzer(system)
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
        n_surfaces_prist = len(surface_prist_comps)
        n_material1d = len(material1d_comps)
        n_material2d = len(material2d_comps)
        n_material2d_prist = len(material2d_prist_comps)
        n_unknown = len(unknown_comps)

        if (n_atoms == 1) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 0):
            return Atom(atoms=atom_comps)

        elif (n_atoms == 0) and \
           (n_molecules == 1) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 0):
            return Molecule(molecules=molecule_comps)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 1) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 0):
            return Material1D(material1d=material1d_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 1) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 0):
            return Material2D(material2d=material2d_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 1) and \
           (n_surfaces_prist == 0):
            return Material2DPristine(material2d_prist=material2d_prist_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 1) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 0):
            return Surface(surfaces=surface_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 0) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 1):
            return SurfacePristine(surfaces_pristine=surface_prist_comps, vacuum_dir=vacuum_dir)

        elif (n_atoms == 0) and \
           (n_molecules == 0) and \
           (n_crystals == 1) and \
           (n_material1d == 0) and \
           (n_material2d == 0) and \
           (n_unknown == 0) and \
           (n_surfaces == 0) and \
           (n_material2d_prist == 0) and \
           (n_surfaces_prist == 0):
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
