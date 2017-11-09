from __future__ import absolute_import, division, print_function, unicode_literals

from systax.analysis.surfaceanalyzer import SurfaceAnalyzer

__metaclass__ = type


class Component():

    def __init__(self, indices, atoms):
        self.indices = indices
        self.atoms = atoms


class SurfaceComponent(Component):

    def __init__(self, indices, atoms, bulk_analyzer, unit_collection):
        super().__init__(indices, atoms)
        self.bulk_analyzer = bulk_analyzer
        self.analyzer = SurfaceAnalyzer(component=self)
        self.unit_collection = unit_collection


class MoleculeComponent(Component):
    pass


class AtomComponent(Component):
    pass


class CrystalComponent(Component):
    def __init__(self, indices, atoms, analyzer=None):
        super().__init__(indices, atoms)
        self.analyzer = SurfaceAnalyzer(component=self)


class Material1DComponent(Component):
    pass


class Material2DComponent(Component):
    pass


class UnknownComponent(Component):
    pass
