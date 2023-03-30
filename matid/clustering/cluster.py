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

    def __len__(self):
        return len(self.indices)

    @lru_cache(maxsize=1)
    def cell(self) -> int:
        """Used to fetch the prototypical cell for this cluster if one exists.
        """
        if self._cell:
            return self._cell

        # When there are multiple regions, return the cell of the region that
        # contains more atoms. TODO: Ultimately a cluster should only have a
        # single region and cell, but currently when clusters are merged there
        # is no mechanism for creating a merged cell or region.
        if self.regions:
            sorted_regions = sorted(self.regions, key=lambda x: -1 if x is None else len(x.get_basis_indices()))
            if sorted_regions[-1] is not None:
                return sorted_regions[0].cell
        return None

    @lru_cache(maxsize=1)
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

    @lru_cache(maxsize=1)
    def classification(self, cluster_threshold=constants.CLUSTER_THRESHOLD) -> str:
        """Used to classify this cluster.
        """
        if self._classification:
            return self._classification

        # Check in how many directions the region is connected to itself.
        n_connected_directions = None
        if self.regions is not None and self.regions[0] is not None:
            best_region = self.regions[0]
            region_conn = best_region.get_connected_directions()
            n_connected_directions = np.sum(region_conn)

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
            if n_connected_directions == 2:
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