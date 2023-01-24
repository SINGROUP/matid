from collections import defaultdict

import numpy as np
from ase.data import covalent_radii

import matid.geometry
from matid.data import constants
from matid.classifications import Class0D, Class1D, Class2D, Class3D, Surface, Material2D, Atom, Unknown
from matid.classification.periodicfinder import PeriodicFinder


class Cluster():
    """
    Represents a part of a bigger system.
    """
    def __init__(self, system, indices, regions, dist_matrix_radii_mic, species=None):
        self.system = system
        self.indices = indices
        self.regions = regions
        self.species = species
        self.merged = False
        self.dist_matrix_radii_pbc = dist_matrix_radii_mic[indices, indices]

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
        dimensionality = matid.geometry.get_dimensionality(
            self.system,
            cluster_threshold,
            self.dist_matrix_radii_pbc
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

            return Cluster(system, final_indices, a.regions + b.regions, self.dist_matrix_radii_mic, target.species), Cluster(system, remainder_indices, source.regions, self.dist_matrix_radii_mic, remainder_species)

        isolated_clusters = []
        while (True):
            if len(clusters) == 0 or clusters[0].merged:
                break
            i_cluster = clusters.pop(0)
            i_indices = i_cluster.indices

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
        # Calculate the displacements here once.
        disp_tensor_mic, disp_factors, disp_tensor_finite, dist_matrix_radii_mic = self.get_displacements(system)
        self.dist_matrix_radii_mic = dist_matrix_radii_mic

        # Iteratively search for new clusters until whole system is covered
        periodic_finder = PeriodicFinder(angle_tol=angle_tol)
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
                disp_tensor_mic=disp_tensor_mic,
                disp_factors=disp_factors,
                disp_tensor_finite=disp_tensor_finite,
                dist_matrix_radii_mic=dist_matrix_radii_mic,
            )
            i_indices = [i_seed]
            if i_grain is not None:
                for unit in i_grain.values():
                    i_valid_indices = [x for x in unit.basis_indices if x is not None]
                    i_indices.extend(i_valid_indices)
            i_species = set(atomic_numbers[i_indices])
            i_indices = set(i_indices)

            clusters.append(Cluster(system, i_indices, [i_grain], self.dist_matrix_radii_mic, i_species))

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

    def get_displacements(self, system):
        """Return the necessary displacement information.

        Args: 
            system (ase.Atoms): The system to investigate.

        Returns:
            A tuple containing all necessary displacement infomration.

        """
        pos = system.get_positions()
        cell = system.get_cell()
        pbc = system.get_pbc()
        disp_tensor_finite = matid.geometry.get_displacement_tensor(pos, pos)
        if pbc.any():
            disp_tensor_mic, disp_factors = matid.geometry.get_displacement_tensor(
                pos,
                pos,
                cell,
                pbc,
                mic=True,
                return_factors=True
            )
        else:
            disp_tensor_mic = disp_tensor_finite
            disp_factors = np.zeros(disp_tensor_finite.shape)
        dist_matrix_mic = np.linalg.norm(disp_tensor_mic, axis=2)

        # Calculate the distance matrix where the periodicity and the covalent
        # radii have been taken into account
        dist_matrix_radii_mic = np.array(dist_matrix_mic)
        num = system.get_atomic_numbers()
        radii = covalent_radii[num]
        radii_matrix = radii[:, None] + radii[None, :]
        dist_matrix_radii_mic -= radii_matrix

        return (disp_tensor_mic, disp_factors, disp_tensor_finite, dist_matrix_radii_mic)

