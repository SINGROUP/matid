import pytest

from numpy.random import default_rng
import numpy as np
import ase.io
from ase.build import surface as ase_surface, bulk
from ase.visualize import view

from matid.clustering import Clusterer, Cluster, Classification
rng = default_rng(seed=7)


def rattle(atoms, displacement=0.08):
    noise = rng.random((len(atoms), 3)) - 0.5
    lengths = np.linalg.norm(noise, axis=1)
    noise /= np.expand_dims(lengths, 1)
    noise *= displacement
    atoms_copy = atoms.copy()
    atoms_copy.set_positions(atoms_copy.get_positions() + noise)
    return atoms_copy


def surface(conv_cell, indices, layers=[3, 3, 2], vacuum=10):
    surface = ase_surface(conv_cell, indices, layers[2], vacuum=vacuum, periodic=True)
    surface *= [layers[0], layers[1], 1]
    return surface


def stack(a, b):
    stacked = ase.build.stack(a, b, axis=2, distance=3, maxstrain=6.7)
    stacked.set_pbc([True, True, False])
    ase.build.add_vacuum(stacked, 10)
    return stacked


surface_fcc_pristine = surface(bulk("Cu", "fcc", a=3.6, cubic=True), [1, 0, 0], vacuum=10)
surface_fcc_noisy = rattle(surface_fcc_pristine)
surface_rocksalt_pristine = surface(bulk("NaCl", "rocksalt", a=5.64, cubic=True), [1, 0, 0], vacuum=10)
surface_rocksalt_noisy = rattle(surface_rocksalt_pristine)
surface_fluorite_pristine = surface(bulk("CaF2", "fluorite", a=5.451), [1, 0, 0], vacuum=10)
surface_fluorite_noisy = rattle(surface_fluorite_pristine)


@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(
        surface_fcc_pristine,
        [Cluster(range(len(surface_fcc_pristine)), dimensionality=2, classification=Classification.Surface)],
        id="fcc surface, pristine"
    ),
    pytest.param(
        surface_fcc_noisy,
        [Cluster(range(len(surface_fcc_noisy)), dimensionality=2, classification=Classification.Surface)],
        id="fcc surface, noisy"
    ),
    pytest.param(
        surface_rocksalt_pristine,
        [Cluster(range(len(surface_rocksalt_pristine)), dimensionality=2, classification=Classification.Surface)],
        id="rocksalt surface, pristine"
    ),
    pytest.param(
        surface_rocksalt_noisy,
        [Cluster(range(len(surface_rocksalt_noisy)), dimensionality=2, classification=Classification.Surface)],
        id="rocksalt surface, noisy"
    ),
    pytest.param(
        surface_fluorite_pristine,
        [Cluster(range(len(surface_fluorite_pristine)), dimensionality=2, classification=Classification.Surface)],
        id="fluorite surface, pristine"
    ),
    pytest.param(
        surface_fluorite_noisy,
        [Cluster(range(len(surface_fluorite_noisy)), dimensionality=2, classification=Classification.Surface)],
        id="fluorite surface, noisy"
    ),
])
def test_clusters(system, clusters_expected):
    # results = Clusterer().get_clusters(system, angle_tol=20, max_cell_size=5, pos_tol=0.4, merge_threshold=0.5, merge_radius=5)
    results = Clusterer().get_clusters(system)
    # view(system)
    # for cluster in results:
    #     indices = list(cluster.indices)
    #     if len(indices) > 1:
    #         cluster_atoms = system[indices]
    #         view(cluster_atoms)

    # Check that correct clusters are found
    assert len(clusters_expected) == len(results)
    cluster_map = {tuple(sorted(x.indices)): x for x in results}
    for cluster_expected in clusters_expected:
        cluster = cluster_map[tuple(sorted(cluster_expected.indices))]
        assert cluster.dimensionality() == cluster_expected.dimensionality()
        assert cluster.classification() == cluster_expected.classification()