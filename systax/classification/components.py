from __future__ import absolute_import, division, print_function, unicode_literals

from systax.analysis.surfaceanalyzer import SurfaceAnalyzer

__metaclass__ = type


class Component():

    def __init__(self, indices, atoms):
        self.indices = indices
        self.atoms = atoms


class SurfaceComponent(Component):

    def __init__(self, indices, atoms, bulk_analyzer, n_layers):
        super().__init__(indices, atoms)
        self.bulk_analyzer = bulk_analyzer
        self.analyzer = SurfaceAnalyzer(component=self)
        self.n_layers = n_layers


class MoleculeComponent(Component):
    pass


class AtomComponent(Component):
    pass


class CrystalComponent(Component):
    pass


class Material1DComponent(Component):
    pass


class Material2DComponent(Component):
    pass


class UnknownComponent(Component):
    pass
