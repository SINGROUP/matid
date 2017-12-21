from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


import itertools

from collections import defaultdict

import numpy as np

import ase.build
from ase import Atoms
from ase.data import covalent_radii
from ase.visualize import view

from systax.exceptions import ClassificationError, SystaxError
from systax.classification.components import ComponentType, Component

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
            max_cell_size=5,
            pos_tol=1,
            vacuum_threshold=6,
            crystallinity_threshold=0.25,
            connectivity_crystal=3.0,
            tesselation_distance=6
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
        self.tesselation_distance = tesselation_distance
        self._repeated_system = None
        self._analyzed = False
        self._orthogonal_dir = None
        self.decisions = {}

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
        dim_class = None
        components = defaultdict(list)
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

        # Get the system dimensionality
        try:
            dimensionality, vacuum_dir = systax.geometry.get_dimensionality(
                system,
                self.connectivity_crystal,
                self.vacuum_threshold,
                disp_tensor=disp_tensor,
                disp_tensor_pbc=disp_tensor_pbc
            )
        except SystaxError:
            components[ComponentType.Unknown].append(Component(np.arange(len(system)), system.copy()))
            return Unknown(components=components)

        # 0D structures
        if dimensionality == 0:
            dim_class = Class0D

            # Check if system consists of one atom or one molecule
            n_atoms = len(system)
            if n_atoms == 1:
                components[ComponentType.Atom].append(Component([0], system))
            else:
                formula = system.get_chemical_formula()
                try:
                    ase.build.molecule(formula)
                except KeyError:
                    components[ComponentType.Unknown].append(Component(np.arange(len(system)), system.copy()))
                else:
                    components[ComponentType.Molecule].append(Component(np.arange(len(system)), system.copy()))

        # 1D structures
        elif dimensionality == 1:

            dim_class = Class1D

            # Find out the dimensions of the system
            is_small = True
            dimensions = systax.geometry.get_dimensions(system, vacuum_dir)
            for i, has_vacuum in enumerate(vacuum_dir):
                if has_vacuum:
                    dimension = dimensions[i]
                    if dimension > 15:
                        is_small = False

            if is_small:
                components[ComponentType.Material1D].append(Component(np.arange(len(system)), system.copy()))
            else:
                components[ComponentType.Unknown].append(Component(np.arange(len(system)), system.copy()))

        # 2D structures
        elif dimensionality == 2:

            # Run the region detection on the whole system.
            periodicfinder = PeriodicFinder(
                pos_tol=self.pos_tol,
                seed_algorithm="cm",
                max_cell_size=self.max_cell_size)
            regions = periodicfinder.get_regions(system, vacuum_dir, self.tesselation_distance)

            # If more than one region found, categorize as unknown
            n_regions = len(regions)
            if n_regions != 1:
                classification = Class2D()
            else:
                region = regions[0]

                # If the region covers less than 50% of the whole system,
                # categorize as Class2D
                n_region_atoms = len(region.get_basis_indices())
                n_atoms = len(system)
                coverage = n_region_atoms/n_atoms

                if coverage < 0.5:
                    classification = Class2D()
                else:
                    # Get information about defects, adsorbates and uncategorized
                    # atoms.
                    adsorbates = region.get_adsorbates()
                    substitutions = region.get_substitutions()
                    vacancies = region.get_vacancies()
                    interstitials = region.get_interstitials()
                    unknowns = region.get_unknowns()

                    if region.is_2d:
                        classification = Material2D(
                            region,
                            adsorbates=adsorbates,
                            substitutions=substitutions,
                            vacancies=vacancies,
                            interstitials=interstitials,
                            unknowns=unknowns,
                        )
                    else:
                        classification = Surface(
                            region,
                            adsorbates=adsorbates,
                            substitutions=substitutions,
                            vacancies=vacancies,
                            interstitials=interstitials,
                            unknowns=unknowns,
                        )

        # Bulk structures
        elif dimensionality == 3:

            dim_class = Class3D

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
                        pos_tol=self.pos_tol,
                        seed_algorithm="cm",
                        max_cell_size=self.max_cell_size)
                    regions = periodicfinder.get_regions(system, vacuum_dir, self.tesselation_distance)

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

            elif is_crystal:
                components[ComponentType.CrystalPristine].append(Component(
                    np.arange(len(system)),
                    system.copy(),
                    analyzer))
            else:
                dim_class = Class3DDisordered

        if classification is not None:
            return classification

        # Return a classification for this system.
        n_molecules = len(components[ComponentType.Molecule])
        n_atoms = len(components[ComponentType.Atom])
        n_crystals_prist = len(components[ComponentType.CrystalPristine])
        n_crystals_defected = len(components[ComponentType.CrystalDefected])
        n_material1d = len(components[ComponentType.Material1D])
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
