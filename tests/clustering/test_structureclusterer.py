import pytest

from ase.build import surface, bulk

from matid import StructureClusterer
from matid.clustering.cluster import Cluster


def rattle(atoms, std=0.05):
    atoms_copy = atoms.copy()
    atoms_copy.rattle(std, seed=7)
    return atoms_copy


surface_fcc_cu_pristine = surface(bulk("Cu", "fcc", a=3.6, cubic=True), [1, 0, 0], 2, vacuum=10, periodic=True)
surface_fcc_cu_noisy = rattle(surface_fcc_cu_pristine)


@pytest.mark.parametrize("system, clusters", [
    pytest.param(
        surface_fcc_cu_pristine,
        [Cluster(surface_fcc_cu_pristine, range(len(surface_fcc_cu_pristine)))],
        id="surface, pristine"
    ),
    pytest.param(
        surface_fcc_cu_noisy,
        [Cluster(surface_fcc_cu_noisy, range(len(surface_fcc_cu_noisy)))],
        id="surface, noise=0.1"
    ),
])
def test_clusters(system, clusters):
    results = StructureClusterer().get_clusters(system)

    # Check that correct clusters are found
    assert len(clusters) == len(results)
    cluster_map = {tuple(sorted(x.indices)): x for x in results}
    for cluster in clusters:
        cluster_map[tuple(sorted(cluster.indices))]


