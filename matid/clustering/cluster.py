import numpy as np

import matid.geometry
from matid.data import constants
from matid.classifications import (
    Class0D, Class1D, Class2D, Class3D, Surface, Material2D, Atom, Unknown
)


class Cluster():
    """
    Represents a part of a bigger system.
    """
    def __init__(self, system=None, indices=None, regions=None, dist_matrix_radii_mic=None, species=None):
        self.system = system
        self.indices = indices
        self.regions = regions
        self.species = species
        self.merged = False
        self.dist_matrix_radii_pbc = dist_matrix_radii_mic

    def classify(self, cluster_threshold=constants.CLUSTER_THRESHOLD):
        """Used to classify this cluster.
        """
        # Check that the region was connected cyclically in two
        # directions. This ensures that finite systems or systems
        # with a dislocation at the cell boundary are filtered.
        best_region = self.regions[0]
        region_conn = best_region.get_connected_directions()
        n_region_conn = np.sum(region_conn)
        region_is_periodic = n_region_conn == 2

        # This might be unnecessary because the connectivity of the
        # unit cell is already checked.
        clusters = best_region.get_clusters()
        basis_indices = set(list(best_region.get_basis_indices()))
        split = True
        for cluster in clusters:
            if basis_indices.issubset(cluster):
                split = False

        # Get the system dimensionality
        classification = Unknown()
        indices = list(self.indices)
        dimensionality = matid.geometry.get_dimensionality(
            self.system[self.indices],
            cluster_threshold,
            self.dist_matrix_radii_pbc[indices, indices]
        )

        # 0D structures
        if dimensionality == 0:
            classification = Class0D()

            # Systems with one atom have their own classification.
            n_atoms = len(self.indices)
            if n_atoms == 1:
                classification = Atom()

        # 1D structures
        elif dimensionality == 1:
            classification = Class1D()

        # 2D structures
        elif dimensionality == 2:
            if not split and region_is_periodic:
                if best_region.is_2d:
                    classification = Material2D()
                else:
                    classification = Surface()
            else:
                classification = Class2D()

        # Bulk structures
        elif dimensionality == 3:
            classification = Class3D()

        return classification