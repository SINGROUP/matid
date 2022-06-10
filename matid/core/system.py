from ase import Atoms
import matid.geometry
import numpy as np


class System(Atoms):

    def __init__(
            self,
            symbols=None,
            positions=None,
            numbers=None,
            tags=None,
            momenta=None,
            masses=None,
            magmoms=None,
            charges=None,
            scaled_positions=None,
            cell=None,
            pbc=None,
            celldisp=None,
            constraint=None,
            calculator=None,
            info=None,
            wyckoff_letters=None,
            equivalent_atoms=None):

        super(System, self).__init__(
            symbols,
            positions,
            numbers,
            tags,
            momenta,
            masses,
            magmoms,
            charges,
            scaled_positions,
            cell,
            pbc,
            celldisp,
            constraint,
            calculator,
            info)

        self.wyckoff_letters = wyckoff_letters
        self.equivalent_atoms = equivalent_atoms

    @staticmethod
    def from_atoms(atoms):
        """Creates a System object from ASE.Atoms object.
        """
        system = System(
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )
        return system

    def to_scaled(self, positions, wrap=False):
        """Used to transform a set of positions to the basis defined by the
        cell of this system.

        Args:
            positions (numpy.ndarray): The positions to scale
            wrap (numpy.ndarray): Whether the positions should be wrapped
                inside the cell.

        Returns:
            numpy.ndarray: The scaled positions
        """
        return matid.geometry.to_scaled(
            self.get_cell(),
            positions,
            wrap,
            self.get_pbc())

    def to_cartesian(self, scaled_positions, wrap=False):
        """Used to transofrm a set of relative positions to the cartesian basis
        defined by the cell of this system.

        Args:
            positions (numpy.ndarray): The positions to scale
            wrap (numpy.ndarray): Whether the positions should be wrapped
                inside the cell.

        Returns:
            numpy.ndarray: The cartesian positions
        """
        return matid.geometry.to_cartesian(
            self.get_cell(),
            scaled_positions,
            wrap,
            self.get_pbc())

    def translate(self, translation, relative=False):
        """Translates the positions by the given translation.

        Args:
            translation (1x3 numpy.array): The translation to apply.
            relative (bool): True if given translation is relative to cell
                vectors.
        """
        matid.geometry.translate(self, translation, relative)

    def get_wyckoff_letters(self):
        """Returns a list of Wyckoff letters for the atoms in the system. This
        information is only available is explicitly set.

        Returns:
            np.ndarray: Wyckoff letters as a list of strings.
        """
        return np.array(self.wyckoff_letters)

    def set_wyckoff_letters(self, wyckoff_letters):
        """Used to set the Wyckoff letters of for the atoms in this system.

        Args:
            wyckoff_letters(sequence of str): The Wyckoff letters for the atoms
            in this system.
        """
        self.wyckoff_letters = np.array(wyckoff_letters)

    def get_equivalent_atoms(self):
        """Returns a list of indices marking the equivalence for the atoms in
        the system. This information is only available is explicitly set.

        Returns:
            np.ndarray: The equivalence information as a list of integers,
            where the same integer means equivalence and an integer is given
            for each atom.
        """
        return np.array(self.equivalent_atoms)

    def set_equivalent_atoms(self, equivalent_atoms):
        """Used to set the list of indices marking the equivalence for the
        atoms for the atoms in this system.

        Args:
            equivalent_atoms(sequence of int): list of indices marking the
                equivalence for the atoms the atoms in this system.
        """
        self.equivalent_atoms = np.array(equivalent_atoms)
