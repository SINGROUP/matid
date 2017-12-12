from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


class Classification():

    def __init__(
            self,
            components,
            vacuum_dir=None,
            analyzer=None
            ):
        self.components = components
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


class SurfaceDefected(Class2D):
    """
    """


class SurfaceAdsorption(Class2D):
    """
    """


class Material2DPristine(Class2D):
    """Consists of one Material2D component without defects or adsorbents.
    """


class Material2DDefected(Class2D):
    """Defected Material2D.
    """


class Material2DAdsorption(Class2D):
    """Adsorption on 2D material.
    """


#===============================================================================
# 3D Structures
class Class3D(Classification):
    """All structures that periodically extend infinitely without vacuum gaps.
    """


class Class3DDisordered(Class3D):
    """All structures that periodically extend infinitely without vacuum gaps.
    """


class CrystalDefected(Class3D):
    """
    """


class CrystalPristine(Classification):
    """
    """


#===============================================================================
# Unknown structures
class Unknown(Classification):
    """
    """
