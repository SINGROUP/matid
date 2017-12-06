from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


import itertools

from collections import defaultdict

import numpy as np

import ase.build
from ase import Atoms
from ase.data import covalent_radii
from ase.visualize import view

from systax.exceptions import ClassificationError
from systax.classification.components import ComponentType, Component

from systax.classification.classifications import \
    SurfacePristine, \
    SurfaceDefected, \
    SurfaceAdsorption, \
    Atom, \
    Molecule, \
    CrystalPristine, \
    CrystalDefected, \
    Material1D, \
    Material2DPristine, \
    Material2DDefected, \
    Material2DAdsorption, \
    Unknown, \
    Class0D, \
    Class1D, \
    Class2D, \
    Class3D, \
    Class3DDisordered, \
    Class3DDisconnected
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
            vacuum_threshold=7,
            crystallinity_threshold=0.25,
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
            vacuum_threshold(float): Amount of vacuum that is considered to
                decouple interaction between structures.
            crystallinity_threshold(float): The threshold of number of symmetry
                operations per atoms in primitive cell that is required for
                crystals.
            connectivity_crystal(float): A parameter that controls the
                connectivity that is required for the atoms of a crystal.
        """
        self.seed_algorithm = seed_algorithm
        self.max_cell_size = max_cell_size
        self.pos_tol = pos_tol
        self.vacuum_threshold = vacuum_threshold
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
        dim_class = None
        components = defaultdict(list)

        # Run a high level analysis to determine the system type
        periodicity = system.get_pbc()
        n_periodic = np.sum(periodicity)

        if n_periodic == 0:
            dim_class = Class0D
            # Check if system has only one atom
            n_atoms = len(system)
            if n_atoms == 1:
                components[ComponentType.Atom].append(Component([0], system))
            # Check if the the system has one or multiple components
            else:
                clusters = systax.geometry.get_clusters(system)
                for cluster in clusters:
                    cluster_atoms = system[cluster]
                    n_atoms_cluster = len(cluster)
                    if n_atoms_cluster == 1:
                        components[ComponentType.Atom].append(Component(cluster, system[cluster]))
                    elif n_atoms_cluster > 1:
                        formula = cluster_atoms.get_chemical_formula()
                        try:
                            ase.build.molecule(formula)
                        except KeyError:
                            components[ComponentType.Unknown].append(Component(cluster, cluster_atoms))
                        else:
                            components[ComponentType.Molecule].append(Component(cluster, cluster_atoms))

        # If the system has at least one periodic dimension, check the periodic
        # directions for a vacuum gap.
        else:

            # Find out the the eigenvectors and eigenvalues of the inertia
            # tensor for an extended version of this system.
            # extended_system = geometry.get_extended_system(system, 15)
            # eigval, eigvec = geometry.get_moments_of_inertia(extended_system)
            # print(eigval)
            # print(eigvec)
            vacuum_dir = systax.geometry.find_vacuum_directions(system, self.vacuum_threshold)
            n_vacuum = np.sum(vacuum_dir)

            # If all directions have a vacuum separating the copies, the system
            # represents a finite structure.
            if n_vacuum == 3:

                dim_class = Class0D

                # Check if system has only one atom
                n_atoms = len(system)
                if n_atoms == 1:
                    components[ComponentType.Atom].append(Component([0], system))

                # Check if the the system has one or multiple components.
                else:
                    clusters = systax.geometry.get_clusters(system)
                    for cluster in clusters:
                        n_atoms_cluster = len(cluster)
                        clust_atoms = system[cluster]
                        if n_atoms_cluster == 1:
                            components[ComponentType.Atom].append(Component(cluster, clust_atoms))
                        elif n_atoms_cluster > 1:
                            formula = clust_atoms.get_chemical_formula()
                            try:
                                ase.build.molecule(formula)
                            except KeyError:
                                components[ComponentType.Unknown].append(Component(cluster, clust_atoms))
                            else:
                                components[ComponentType.Molecule].append(Component(cluster, clust_atoms))

            # 1D structures
            if n_vacuum == 2:

                dim_class = Class1D

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
                    components[ComponentType.Material1D].append(Component(np.arange(len(system)), system.copy()))
                else:
                    components[ComponentType.Unknown].append(Component(np.arange(len(system)), system.copy()))

            # 2D structures
            if n_vacuum == 1:

                dim_class = Class2D

                # Run the region detection on the whole system.
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
                    components[ComponentType.Unknown].append(Component(list(i_misc_ind), system[list(i_misc_ind)]))

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

                        # print(layer_mean)
                        # print(layer_std)

                        if layer_mean == self.layers_2d and thickness < self.thickness_2d:
                            if is_pristine:
                                components[ComponentType.Material2DPristine].append(Component(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                            else:
                                components[ComponentType.Material2DDefected].append(Component(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                        elif layer_mean > self.layers_2d:
                            if is_pristine:
                                components[ComponentType.SurfacePristine].append(Component(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                            else:
                                components[ComponentType.SurfaceDefected].append(Component(i_orig_indices, i_atoms, i_unit_collection, analyzer_2d))
                    else:
                        components[ComponentType.Unknown].append(Component(np.arange(len(system)), system.copy()))

            # Bulk structures
            if n_vacuum == 0:

                dim_class = Class3D

                # Check the number of symmetries
                analyzer = Class3DAnalyzer(system)
                is_crystal = check_if_crystal(analyzer, threshold=self.crystallinity_threshold)

                # Check the number of clusters
                repetitions = [2, 2, 2]
                ext_sys3d = system.repeat(repetitions)
                clusters = systax.geometry.get_clusters(ext_sys3d, self.connectivity_crystal)
                n_clusters = len(clusters)

                # If the structure is connected but the symmetry criteria was
                # not fullfilled, check the number of atoms in the primitive
                # cell. If above a certain threshold, try to find periodic
                # region to see if it is a crystal containing a defect.
                if not is_crystal and n_clusters == 1:
                    primitive_system = analyzer.get_primitive_system()
                    n_atoms_prim = len(primitive_system)
                    if n_atoms_prim >= 20:
                        periodicfinder = PeriodicFinder(
                            pos_tol=self.pos_tol,
                            seed_algorithm="cm",
                            max_cell_size=self.max_cell_size)
                        regions = periodicfinder.get_regions(system, vacuum_dir)

                        # If all the regions cover at least 80% of the structure,
                        # then we consider it to be a defected crystal
                        n_atoms_in_regions = 0
                        n_atoms_total = len(system)
                        for region in regions:
                            n_atoms_in_regions += len(region[0])
                        coverage = n_atoms_in_regions/n_atoms_total
                        if coverage >= 0.8:
                            dim_class = CrystalDefected
                        else:
                            dim_class = Class3DDisordered

                elif is_crystal and n_clusters == 1:
                    components[ComponentType.CrystalPristine].append(Component(
                        np.arange(len(system)),
                        system.copy(),
                        analyzer))
                elif n_clusters > 1:
                    dim_class = Class3DDisconnected
                else:
                    dim_class = Class3DDisordered

        # Return a classification for this system.
        n_molecules = len(components[ComponentType.Molecule])
        n_atoms = len(components[ComponentType.Atom])
        n_crystals_prist = len(components[ComponentType.CrystalPristine])
        n_crystals_defected = len(components[ComponentType.CrystalDefected])
        n_surfaces_prist = len(components[ComponentType.SurfacePristine])
        n_surfaces_defected = len(components[ComponentType.SurfaceDefected])
        n_material1d = len(components[ComponentType.Material1D])
        n_material2d_prist = len(components[ComponentType.Material2DPristine])
        n_material2d_defected = len(components[ComponentType.Material2DDefected])
        n_types = 0
        for item in components.values():
            if len(item) != 0:
                n_types += 1

        if n_types == 1:
            if n_atoms == 1:
                return Atom(components=components)
            elif n_molecules == 1:
                return Molecule(components=components)
            elif n_material1d == 1:
                return Material1D(components=components)
            elif n_material2d_prist == 1:
                return Material2DPristine(components=components)
            elif n_material2d_defected == 1:
                return Material2DDefected(components=components)
            elif n_surfaces_prist == 1:
                return SurfacePristine(components=components)
            elif n_surfaces_defected == 1:
                return SurfaceDefected(components=components)
            elif n_crystals_prist == 1:
                return CrystalPristine(components=components)
            elif n_crystals_defected == 1:
                return CrystalDefected(components=components)
            else:
                return dim_class(components=components)
        else:
            return dim_class(
                components=components
            )
