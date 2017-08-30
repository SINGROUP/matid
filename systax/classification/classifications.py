from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


class Classification():

    def __init__(self, surfaces=None, atoms=None, molecules=None, crystals=None):
        self.surfaces = surfaces
        self.atoms = atoms
        self.molecules = molecules
        self.crystals = crystals


class Atom(Classification):
    """
    """


class Molecule(Classification):
    """
    """


class Surface(Classification):
    """
    """


class Crystal(Classification):
    """
    """


class Material1D(Classification):
    """
    """


class Material2D(Classification):
    """
    """


class AdsorptionSystem(Classification):
    """
    """


class MultiComponent(Classification):
    """
    """


class Unknown(Classification):
    """
    """
