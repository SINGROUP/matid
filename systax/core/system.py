from __future__ import absolute_import, division, print_function, unicode_literals

from ase import Atoms
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
        fractional = np.linalg.solve(
            self.get_cell(complete=True).T,
            positions.T).T

        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    # See the scaled_positions.py test.
                    fractional[:, i] %= 1.0
                    fractional[:, i] %= 1.0

        return fractional

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
        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    # See the scaled_positions.py test.
                    scaled_positions[:, i] %= 1.0
                    scaled_positions[:, i] %= 1.0

        cartesian_positions = scaled_positions.dot(self.get_cell().T)
        return cartesian_positions

    def translate(self, translation, relative=False):
        """Translates the positions by the given translation.

        Args:
            translation (1x3 numpy.array): The translation to apply.
            relative (bool): True if given translation is relative to cell
                vectors.
        """
        if relative:
            rel_pos = self.get_scaled_positions()
            rel_pos += translation
            self.set_scaled_positions(rel_pos)
        else:
            cart_pos = self.get_positions()
            cart_pos += translation
            self.set_positions(cart_pos)

    def get_wyckoff_letters(self):
        """Returns a list of Wyckoff letters for the atoms in the system. This
        information is only available is explicitly set.

        Returns:
            np.ndarray: Wyckoff letters as a list of strings.
        """
        return np.array(self.wyckoff_letters)

    def set_wyckoff_letters(self, wyckoff_letters):
        self.wyckoff_letters = np.array(wyckoff_letters)
