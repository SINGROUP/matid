from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ase import Atoms

import systax.geometry


class LinkedUnitCollection(dict):
    """Represents a collection of similar cells that are connected in 3D space
    to form a structure, e.g. a surface.

    Essentially this is a special flavor of a regular dictionary: the keys can
    only be a sequence of three integers, and the values should be LinkedUnits.
    """
    def __init__(self, system, cell, is_2d, vacuum_gaps, delaunay_threshold):
        """
        Args:
            system(ase.Atoms): A reference to the system from which this
            LinkedUniCollection is gathered.
        """
        self.system = system
        self.cell = cell
        self.is_2d = is_2d
        self.vacuum_gaps = vacuum_gaps
        self.delaunay_threshold = delaunay_threshold
        self._decomposition = None
        self._inside_indices = None
        self._outside_indices = None
        self._adsorbates = None
        self._substitutions = None
        self._vacancies = None
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

    def get_basis_indices(self):
        """Returns the indices of the atoms that were found to belong to a unit
        cell basis in the LinkedUnits in this collection as a single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        indices = set()
        for unit in self.values():
            i_indices = [x for x in unit.basis_indices if x is not None]
            indices.update(i_indices)

        return np.array(list(indices))

    def get_invalid_indices(self):
        """Get the indices of atoms that do not belong to the basis of this
        region.
        """
        all_indices = set(range(len(self.system)))
        basis_indices = set(self.get_basis_indices())
        invalid_indices = all_indices - basis_indices

        return np.array(list(invalid_indices))

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

    def get_adsorbates(self):
        """Get the indices of the adsorbate atoms in the region.

        All atoms that are outside the tesselation, and which are not part of
        the elements present in the surface are labeled as adsorbate atoms.

        This function does not differentiate between different adsorbate
        molecules.

        Returns:
            np.ndarray: Indices of the adsorbates in the original system.
        """
        if self._adsorbates is None:

            _, outside_indices = self.get_inside_and_outside_indices()
            basis_elements = self.cell.get_atomic_numbers()
            num = self.system.get_atomic_numbers()
            if len(outside_indices) != 0:
                outside_num = num[outside_indices]
                adsorbate_mask = ~np.isin(outside_num, basis_elements)
                adsorbates = outside_indices[adsorbate_mask]
            else:
                adsorbates = np.array([])

            self._adsorbates = adsorbates

        return self._adsorbates

    def get_substitutions(self):
        """Get the substitutions in the region.
        """
        if self._substitutions is None:

            # Gather all substitutions
            all_substitutions = []
            for cell in self.values():
                subst = cell.substitutions
                if len(subst) != 0:
                    all_substitutions.extend(subst)

            # In 2D materials all substitutions in the cell are valid
            # substitutions
            if self.is_2d:
                self._substitutions = all_substitutions
            else:
                # In surfaces the substitutions have to be validate by whether they
                # are inside the tesselation or not
                inside_indices, _ = self.get_inside_and_outside_indices()
                inside_set = set(inside_indices)

                # Find substitutions that are inside the tesselation
                valid_subst = []
                for subst in all_substitutions:
                    subst_index = subst.index
                    if subst_index in inside_set:
                        valid_subst.append(subst)
                self._substitutions = valid_subst

        return self._substitutions

    def get_vacancies(self):
        """Get the vacancies in the region.

        Returns:
            ASE.Atoms: An atoms object representing the atoms that are missing.
            The Atoms object has the same properties as the original system.
        """
        if self._vacancies is None:

            # Gather all cavancies
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

    def get_unknowns(self):
        """Get the indices that form an unknown part around the region.
        """
        # Get all indices
        all_indices = set(self.get_all_indices())
        basis_indices = set(self.get_basis_indices())
        interstitials = set(self.get_interstitials())
        adsorbates = set(self.get_adsorbates())
        substitutions = set(self.get_substitutions())
        subst_indices = set([x.index for x in substitutions])

        combined = basis_indices.union(interstitials).union(adsorbates).union(subst_indices)
        unknowns = all_indices - combined
        return np.array(list(unknowns))

    def get_tetrahedra_decomposition(self):
        """Get the tetrahedra decomposition for this region.
        """
        if self._decomposition is None:
            # Get the positions of basis atoms
            basis_indices = self.get_basis_indices()
            valid_sys = self.system[basis_indices]

            # Perform tetrahedra decomposition
            self._decomposition = systax.geometry.get_tetrahedra_decomposition(
                valid_sys,
                self.vacuum_gaps,
                self.delaunay_threshold
            )

        return self._decomposition

    def get_all_indices(self):
        return set(range(len(self.system)))

    def get_inside_and_outside_indices(self):
        """Get the indices of atoms that are inside and outside the tetrahedra
        tesselation.

        Returns:
            (np.ndarray, np.ndarray): Indices of atoms that are inside and
            outside the tesselation. The inside indices are in the first array.
        """
        if self._inside_indices is None and self._outside_indices is None:
            invalid_indices = self.get_invalid_indices()
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
        # self.substitute_indices = substitute_indices


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
