from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


class Classification():

    def __init__(
            self,
            surfaces_prist=None,
            atoms=None,
            molecules=None,
            crystals_prist=None,
            material1d=None,
            material2d_prist=None,
            unknowns=None,
            vacuum_dir=None,
            analyzer=None
            ):
        self.surfaces_prist = surfaces_prist
        self.atoms = atoms
        self.molecules = molecules
        self.crystals_prist = crystals_prist
        self.material1d = material1d
        self.material2d_prist = material2d_prist
        self.unknowns = unknowns
        self.vacuum_dir = vacuum_dir
        self.analyzer = analyzer


#===============================================================================
# 0D Structures
class Class0D(Classification):
    """Structures that have a structure that is isolated in all directions by a
    vacuum gap.
    """


class Atom(Class0D):
    """
    """


class Molecule(Class0D):
    """
    """


class NanoCluster(Class0D):
    """
    """


#===============================================================================
# 1D Structures
class Class1D(Classification):
    """All structures that are roughly 1-dimensional, meaning that one
    dimension is much larger than the two others.
    """


class Material1D(Class1D):
    """
    """


#===============================================================================
# 2D Structures
class Class2D(Classification):
    """All structures that are roughly 2-dimensional, meaning that two
    dimensions are much larger than the two others.
    """


class SurfacePristine(Class2D):
    """
    """


class Material2DPristine(Class2D):
    """Consists of one Material2D component without defects or adsorbents.
    """


#===============================================================================
# 3D Structures
class Class3D(Classification):
    """All structures that periodically extend infinitely without vacuum gaps.
    """


class CrystalPristine(Classification):
    """
    """


#===============================================================================
# Unknown structures
class Unknown(Classification):
    """
    """
