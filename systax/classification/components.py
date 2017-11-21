from __future__ import absolute_import, division, print_function, unicode_literals

# from systax.analysis.class2danalyzer import Class2DAnalyzer
# from systax.analysis.class3danalyzer import Class3DAnalyzer

__metaclass__ = type


class Component():

    def __init__(self, indices, atoms, unit_collection=None, analyzer=None):
        self.indices = indices
        self.atoms = atoms
        self.analyzer = analyzer
        self.unit_collection = None


class MoleculeComponent(Component):
    pass


class AtomComponent(Component):
    pass


class CrystalComponent(Component):
    pass


class Material1DComponent(Component):
    pass


class SurfaceComponent(Component):
    pass


class SurfacePristineComponent(Component):
    pass


class Material2DComponent(Component):
    pass


class Material2DPristineComponent(Component):
    pass


class UnknownComponent(Component):
    pass
