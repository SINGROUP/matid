import pytest

from ase.build import surface, bulk
from matid.clustering import Clusterer, Cluster, Classification


def rattle(atoms, std=0.05):
    atoms_copy = atoms.copy()
    atoms_copy.rattle(std, seed=7)
    return atoms_copy


surface_fcc_cu_pristine = surface(bulk("Cu", "fcc", a=3.6, cubic=True), [1, 0, 0], 2, vacuum=10, periodic=True)
surface_fcc_cu_noisy = rattle(surface_fcc_cu_pristine)


@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(
        surface_fcc_cu_pristine,
        [Cluster(range(len(surface_fcc_cu_pristine)), dimensionality=2, classification=Classification.Surface)],
        id="surface, pristine"
    ),
    pytest.param(
        surface_fcc_cu_noisy,
        [Cluster(range(len(surface_fcc_cu_noisy)), dimensionality=2, classification=Classification.Surface)],
        id="surface, noise=0.1"
    ),
])
def test_clusters(system, clusters_expected):
    results = Clusterer().get_clusters(system)

    # Check that correct clusters are found
    assert len(clusters_expected) == len(results)
    cluster_map = {tuple(sorted(x.indices)): x for x in results}
    for cluster_expected in clusters_expected:
        cluster = cluster_map[tuple(sorted(cluster_expected.indices))]
        assert cluster.dimensionality() == cluster_expected.dimensionality()
        assert cluster.classification() == cluster_expected.classification()