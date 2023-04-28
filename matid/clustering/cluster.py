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
    def __init__(self, indices=None, species=None, region=None, dimensionality=None, classification=None, cell=None, system=None, distances=None, bond_threshold=None):
        if isinstance(indices, list):
            self.indices = indices
        else:
            self.indices = list(indices)
        self.species = species
        self.region = region
        self._dimensionality = dimensionality
        self._classification = classification
        self._cell = cell
        self.system = system
        self.distances = distances
        self.merged = False
        self.bond_threshold = bond_threshold

    def __len__(self):
        return len(self.indices)

    def _distance_matrix_radii_mic(self) -> int:
        """Used to fetch the prototypical cell for this cluster if one exists.
        """
        return self.distances.dist_matrix_radii_mic[np.ix_(self.indices, self.indices)]

    @lru_cache(maxsize=1)
    def cell(self) -> int:
        """Used to fetch the prototypical cell for this cluster if one exists.
        """
        if self._cell:
            return self._cell
        if self.region:
            return self.region.cell
        return None

    @lru_cache(maxsize=1)
    def dimensionality(self) -> int:
        """Used to fetch the dimensionality of the cluster.
        """
        if self._dimensionality is not None:
            return self._dimensionality

        return matid.geometry.get_dimensionality(
            self.system[self.indices],
            self.bond_threshold,
            dist_matrix_radii_mic_1x=self._distance_matrix_radii_mic()
        )

    @lru_cache(maxsize=1)
    def classification(self) -> str:
        """Used to classify this cluster.
        """
        if self._classification:
            return self._classification

        # Check in how many directions the region is connected to itself.
        n_connected_directions = None
        if self.region is not None:
            region_conn = self.region.get_connected_directions()
            n_connected_directions = np.sum(region_conn)

        # Get the system dimensionality
        dimensionality = self.dimensionality()

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
                if self.region.is_2d:
                    cls = Classification.Material2D
                else:
                    cls = Classification.Surface
            else:
                cls = Classification.Class2D

        # 3D structures
        elif dimensionality == 3:
            cls = Classification.Class3D

        return cls