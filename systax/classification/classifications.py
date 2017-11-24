from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


class Classification():

    def __init__(
            self,
            surfaces=None,
            surfaces_prist=None,
            atoms=None,
            molecules=None,
            crystals=None,
            material1d=None,
            material2d=None,
            material2d_prist=None,
            unknowns=None,
            vacuum_dir=None,
            analyzer=None
            ):
        self.surfaces = surfaces
        self.surfaces_prist = surfaces_prist
        self.atoms = atoms
        self.molecules = molecules
        self.crystals = crystals
        self.material1d = material1d
        self.material2d = material2d
        self.material2d_prist = material2d_prist
        self.unknowns = unknowns
        self.vacuum_dir = vacuum_dir
        self.analyzer = analyzer


class Defected():
    """Structures with any kind of defect.
    """


class Adsorption():
    """Structures that represent adsorption.
    """


#===============================================================================
# 0D Structures
class Class0D(Classification):
    """
    """


class Atom(Class0D):
    """
    """


class Molecule(Class0D):
    """
    """


#===============================================================================
# 1D Structures
class Class1D(Classification):
    """
    """


class Material1D(Class1D):
    """
    """


#===============================================================================
# 2D Structures
class Class2D(Classification):
    """
    """


class Surface(Class2D):
    """
    """


class SurfacePristine(Surface):
    """
    """


class SurfaceAdsorption(Surface, Adsorption):
    """
    """


class Material2D(Class2D):
    """
    """


class Material2DPristine(Material2D):
    """Consists of one Material2D component without defects or adsorbents.
    """


class Material2DAdsorption(Material2D, Adsorption):
    """
    """


#===============================================================================
# 3D Structures
class Class3D(Classification):
    """
    """


class Crystal(Class3D):
    """
    """


class CrystalPristine(Crystal):
    """
    """


class Unknown(Classification):
    """
    """
