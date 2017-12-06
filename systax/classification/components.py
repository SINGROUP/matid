from __future__ import absolute_import, division, print_function, unicode_literals
from enum import Enum


class ComponentType(Enum):

    Atom = 0,
    Molecule = 1,
    CrystalPristine = 2,
    CrystalDefected = 3,
    Material1D = 4,
    Material2DPristine = 5,
    Material2DDefected = 6,
    SurfacePristine = 7,
    SurfaceDefected = 8,
    Unknown = 9,


class Component():

    def __init__(self, indices, atoms, comp_type=None, unit_collection=None, analyzer=None):
        self.indices = indices
        self.atoms = atoms
        self.type = comp_type
        self.analyzer = analyzer
        self.unit_collection = unit_collection
