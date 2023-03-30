from enum import Enum
from functools import lru_cache
import numpy as np

import matid.geometry
from matid.data import constants


class Classification(Enum):
    Unknown = "Unknown"
    Class0D = "0D Generic"
    Class1D = "1D Generic"
    Class2D = "2D Generic"
    Class3D = "3D Generic"
    Atom = "0D Atom"
    Material2D = "2D Material"
    Surface = "2D Surface"


class Cluster():
    """
    Represents a part of a bigger system.
    """
    def __init__(self, indices=None, species=None, regions=None, dimensionality=None, classification=None, cell=None, system=None, distances=None):
        self.indices = indices
        self.species = species
        self.regions = regions
        self._dimensionality = dimensionality
        self._classification = classification
        self._cell = cell
        self.system = system
        self.distances = distances
        self.merged = False

    # @lru_cache(maxsize=1)
    def cell(self) -> int:
        """Used to fetch the prototypical cell for this cluster if one exists.
        """
        if self._cell:
            return self._cell

    # @lru_cache(maxsize=1)
    def dimensionality(self, cluster_threshold=constants.CLUSTER_THRESHOLD) -> int:
        """Used to fetch the dimensionality of the cluster.
        """
        if self._dimensionality:
            return self._dimensionality
        indices = list(self.indices)
        return matid.geometry.get_dimensionality(
            self.system[indices],
            cluster_threshold,
            self.distances.dist_matrix_radii_mic[np.ix_(indices, indices)]
        )

    # @lru_cache(maxsize=1)
    def classification(self, cluster_threshold=constants.CLUSTER_THRESHOLD) -> str:
        """Used to classify this cluster.
        """
        if self._classification:
            return self._classification
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
        dimensionality = self.dimensionality(cluster_threshold)

        # 0D structures
        cls = Classification.Unknown
        if dimensionality == 0:
            cls = Classification.Class0D

            # Systems with one atom have their own classification.
            n_atoms = len(self.indices)
            if n_atoms == 1:
                cls = Classification.Atom

        # 1D structures
        elif dimensionality == 1:
            cls = Classification.Class1D

        # 2D structures
        elif dimensionality == 2:
            if not split and region_is_periodic:
                if best_region.is_2d:
                    cls = Classification.Material2D
                else:
                    cls = Classification.Surface
            else:
                cls = Classification.Class2D

        # 3D structures
        elif dimensionality == 3:
            cls = Classification.Class3D

        return cls