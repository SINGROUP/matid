from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


class Classification():

    def __init__(
            self,
            surfaces=None,
            atoms=None,
            molecules=None,
            crystals=None,
            material1d=None,
            material2d=None,
            unknowns=None,
            vacuum_dir=None,
            analyzer=None
            ):
        self.surfaces = surfaces
        self.atoms = atoms
        self.molecules = molecules
        self.crystals = crystals
        self.material1d = material1d
        self.material2d = material2d
        self.unknowns = unknowns
        self.vacuum_dir = vacuum_dir
        self.analyzer = analyzer


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
