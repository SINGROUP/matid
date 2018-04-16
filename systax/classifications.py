from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import super
__metaclass__ = type


class Classification():
    pass


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
    """Base class for all structures that are roughly 2-dimensional, meaning that two of the
    dimensions are much larger than the remaining one.
    """
    def __init__(self):
        super().__init__()


class Class2DWithCell(Class2D):
    """Two dimensional structures from which a periodic unit cell has been
    identified.
    """
    def __init__(
            self,
            region,
            ):
        super().__init__()
        self.region = region

    @property
    def basis_indices(self):
        return self.region.get_basis_indices()

    @property
    def outliers(self):
        return self.region.get_outliers()

    @property
    def interstitials(self):
        return self.region.get_interstitials()

    @property
    def adsorbates(self):
        return self.region.get_adsorbates()

    @property
    def substitutions(self):
        return self.region.get_substitutions()

    @property
    def vacancies(self):
        return self.region.get_vacancies()

    @property
    def unknowns(self):
        return self.region.get_unknowns()

    @property
    def prototype_cell(self):
        return self.region.cell


class Surface(Class2DWithCell):
    """
    """


class Material2D(Class2DWithCell):
    """
    """


#===============================================================================
# 3D Structures
class Class3D(Classification):
    """All structures that periodically extend infinitely without vacuum gaps.
    """
    pass


class Crystal(Class3D):
    def __init__(
            self,
            cell_analyzer,
            ):
        self.cell_analyzer = cell_analyzer


#===============================================================================
# Unknown structures
class Unknown(Classification):
    """
    """
