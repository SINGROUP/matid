from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import super
__metaclass__ = type


class Classification():

    def __init__(
            self,
            vacuum_dir=None,
            ):
        self.vacuum_dir = vacuum_dir


#===============================================================================
# 0D Structures
class Class0D(Classification):
    """Structures that have a structure that is isolated in all directions by a
    vacuum gap.
    """
    def __init__(self):
        pass


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
    def __init__(
            self,
            vacuum_dir=None,
            region=None,
            cell_analyzer=None
            ):
        super().__init__(vacuum_dir)
        self.region = region
        if region is not None:
            self.basis_indices = region.get_basis_indices()
            self.interstitials = region.get_interstitials()
            self.substitutions = region.get_substitutions()
            self.adsorbates = region.get_adsorbates()
            self.vacancies = region.get_vacancies()
            self.unknowns = region.get_unknowns()
        else:
            self.basis_indices = ()
            self.interstitials = ()
            self.substitutions = ()
            self.adsorbates = ()
            self.vacancies = ()
            self.unknowns = ()
        self.vacuum_dir = vacuum_dir
        self.cell_analyzer = cell_analyzer


class Surface(Class2D):
    """
    """


class Material2D(Class2D):
    """
    """


#===============================================================================
# 3D Structures
class Class3D(Classification):
    """All structures that periodically extend infinitely without vacuum gaps.
    """
    def __init__(
            self,
            cell_analyzer,
            region=None,
            ):
        self.region = region
        if region is not None:
            self.basis_indices = region.get_basis_indices()
            self.interstitials = region.get_interstitials()
            self.substitutions = region.get_substitutions()
            self.vacancies = region.get_vacancies()
            self.unknowns = region.get_unknowns()
        else:
            self.basis_indices = ()
            self.interstitials = ()
            self.substitutions = ()
            self.vacancies = ()
            self.unknowns = ()
        self.cell_analyzer = cell_analyzer


class Crystal(Class3D):
    """
    """


#===============================================================================
# Unknown structures
class Unknown(Classification):
    """
    """
