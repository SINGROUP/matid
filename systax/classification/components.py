class Component(object):

    def __init__(self, indices, atoms):
        self.indices = indices
        self.atoms = atoms


class SurfaceComponent(Component):

    def __init__(self, indices, atoms, bulk_system, symmetry_dataset):
        super().__init__(indices, atoms)
        self.bulk_system = bulk_system
        self.symmetry_dataset = symmetry_dataset

    def get_miller_index(self):
        """
        """

    def get_equivalent_miller_indices(self):
        """
        """

    def get_normalized_cell(self):
        """
        """
        return self.bulk_system

    def get_bulk_primitive_cell(self):
        """
        """

    def get_bulk_symmetry_dataset(self):
        """
        """
        return self.symmetry_dataset


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
