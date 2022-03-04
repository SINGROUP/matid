from collections import defaultdict
import numpy as np
import ase.io
from ase.io import write
from ase.visualize import view
from matid.classification.periodicfinder import PeriodicFinder


class Cluster():
    """
    Represents a part of a bigger system.
    """
    def __init__(self, indices, regions, species=None):
        self.indices = indices
        self.regions = regions
        self.species = species
        self.merged = False


class StructureClusterer():
    """
    Class for partitioning a more complex system into structurally similar
    clusters.

    You can apply this class for e.g. partitioning a larger material into
    grains, a heterostructure into it's component etc. The clustering is based
    on finding periodically repeating motifs, and as such it is not suitable
    for e.g. finding molecules. Any atoms that do not have enough periodic
    repetitions will be returned as isolated clusters.
    """
    def __init__(self, seed=7):
        """
        Args:
            seed(int): The seed that is used for random number generation.
        """
        self.rng = np.random.default_rng(seed)

    def localize_clusters(self, system, clusters, merge_radius):
        """
        Used to resolve overlaps between clusters by assigning the overlapping
        atoms to the "nearest" cluster.

        Args:
            system (ase.Atoms): The original system.
            clusters (list of Clusters): The clusters to localize.
            merge_radius (float): The radius to consider when assigning atoms
              to a particular cluster.

        Returns:
            List of Clusters that no longer have overlapping atoms as each atom
            has been assigned to the nearest cluster.
        """
        # Get all overlapping atoms, and the regions with which they overlap
        overlap_map = defaultdict(list)
        for i in range(len(system)):
            for cluster in clusters:
                if i in cluster.indices:
                    overlap_map[i].append(cluster)

        # Assign each overlapping atom to the cluster that is "nearest". Notice
        # that we do not update the regions during the process.
        positions = system.get_positions()
        for i, i_clusters in overlap_map.items():
            if len(i_clusters) > 1:
                surrounding_indices = np.argwhere(np.linalg.norm(positions - positions[i], axis=1) < merge_radius)[:, 0]
                max_near = 0
                max_cluster = i_clusters[0]
                for cluster in i_clusters:
                    n_near = len(cluster.indices.intersection(surrounding_indices))
                    if n_near > max_near:
                        max_near = n_near
                        max_cluster = cluster
                for cluster in i_clusters:
                    if cluster != max_cluster:
                        cluster.indices.remove(i)
        return clusters

    def merge_clusters(self, system, clusters, merge_threshold):
        """
        Used to merge higly overlapping clusters.
        """
        def merge(system, a, b):
            """
            Merges the given two clusters.
            """
            # If there are conficting species in the regions that are merged, only
            # the species from the larger are kept during the merge. This helps
            # getting rid of artifical regions at the interface between two
            # clusters.
            atomic_numbers = system.get_atomic_numbers()
            if len(a.indices) > len(b.indices):
                target = a
                source = b
            else:
                target = b
                source = a
            common = set(filter(lambda x: atomic_numbers[x] in target.species, source.indices))
            remainder_indices = source.indices - common
            remainder_species = set(atomic_numbers[list(remainder_indices)])
            final_indices = target.indices.union(common)

            return Cluster(final_indices, a.regions + b.regions, target.species), Cluster(remainder_indices, source.regions, remainder_species)

        covered_grains = set()
        new_grain_map = {}
        atomic_numbers = system.get_atomic_numbers()
        remainders = []
        isolated_clusters = []
        while (True):
            if len(clusters) == 0 or clusters[0].merged:
                break
            i_cluster = clusters.pop(0)
            i_indices = i_cluster.indices
            i_region = i_cluster.regions[0]

            # Check overlap with all other non-isolated clusters
            isolated = True
            if len(clusters):
                overlaps = [(j, len(i_indices.intersection(j_cluster.indices))) for j, j_cluster in enumerate(clusters)]
                overlaps = sorted(overlaps, key=lambda x: x[1], reverse=True)
                best_overlap = overlaps[0][1]
                best_grain = overlaps[0][0]
                target_cluster = clusters[best_grain]

                # Find the biggest overlap and if it is large enough, merge these
                # clusters together. Large overlap indicates that they are actually
                # part of the same component.
                best_overlap_score = max(best_overlap/len(i_indices), best_overlap/len(target_cluster.indices))

                if best_overlap_score > merge_threshold:
                    merged, remainder = merge(system, i_cluster, target_cluster)
                    merged.merged = True
                    isolated = False
                    clusters.pop(best_grain)
                    clusters.append(merged)
                    if remainder.indices:
                        clusters.insert(0, remainder)
            # Component without enough overlap is saved as it is.
            if isolated:
                isolated_clusters.append(i_cluster)

        return isolated_clusters + clusters

    def get_clusters(self, system, angle_tol=20, max_cell_size=5, pos_tol=0.25, merge_threshold=0.5, merge_radius=5):
        """
        Used to detect and return structurally separate clusters within the
        given system.

        Args:
            system (ase.Atoms): The structure to partition.
            angle_tol (float): angle_tol parameter for PeriodicFinder
            max_cell_size (float): max_cell_size parameter for PeriodicFinder.get_region
            pos_tol (float): pos_tol parameter for PeriodicFinder.get_region
            merge_threshold (float): A threshold for merging two clusters
              together. Give as a fraction of shared atoms. Value of 1 would
              mean that clusters are never merged, value of 0 means that they
              are merged always when at least one atom is shared.
            merge_radius (float): Radius for finding nearby atoms when deciding
                which cluster is closest. Given in angstroms.

        Returns:
            A list of Clusters.
        """
        # Iteratively search for new clusters until whole system is covered
        periodic_finder = PeriodicFinder(angle_tol=angle_tol)
        covered = False
        indices = set(list(range(len(system))))
        clusters = []
        atomic_numbers = system.get_atomic_numbers()
        while len(indices) != 0:
            i_seed = self.rng.choice(list(indices), 1)[0]
            i_grain = periodic_finder.get_region(
                system,
                seed_index=i_seed,
                max_cell_size=max_cell_size,
                pos_tol=pos_tol,
            )
            i_indices = [i_seed]
            if i_grain is not None:
                for unit in i_grain.values():
                    i_valid_indices = [x for x in unit.basis_indices if x is not None]
                    i_indices.extend(i_valid_indices)
            i_species = set(atomic_numbers[i_indices])
            i_indices = set(i_indices)

            clusters.append(Cluster(i_indices, [i_grain], i_species))

            # Remove found grain from set of indices
            if len(i_indices) == 0:
                i_indices = set([i_seed])
            indices = indices - i_indices

        # Check overlaps of the regions. For large overlaps the grains are
        # merged (the real region was probably cut into pieces by unfortunate
        # selection of the seed atom)
        clusters = self.merge_clusters(system, clusters, merge_threshold)

        # Any remaining overlaps are resolved by assigning atoms to the
        # "nearest" cluster
        clusters = self.localize_clusters(system, clusters, merge_radius)

        return clusters

