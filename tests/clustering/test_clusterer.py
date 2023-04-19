import pytest

from numpy.random import default_rng
import numpy as np
import ase.io
from ase import Atoms
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


def stack(a, b, axis=2, distance=3, vacuum=10):
    a_pos = a.get_positions()[:, axis]
    a_max = np.max(a_pos)
    a_min = np.min(a_pos)
    b_pos = b.get_positions()[:, axis]
    b_max = np.max(b_pos)
    b_min = np.min(b_pos)
    a_shift = np.zeros((len(a), 3))
    a_shift[:, axis] += -a_min
    b_shift = np.zeros((len(b), 3))
    b_shift[:, axis] += -b_min + (a_max-a_min) + distance
    a.translate(a_shift)
    b.translate(b_shift)
    stacked = a + b
    cell = a.get_cell()
    axis_new = cell[axis, :]
    axis_norm = np.linalg.norm(axis_new)
    axis_new = axis_new / axis_norm * (a_max - a_min + b_max - b_min + distance)
    cell[axis, :] = axis_new
    stacked.set_cell(cell)
    ase.build.add_vacuum(stacked, vacuum)
    return stacked


surface_fcc_pristine = surface(bulk("Cu", "fcc", a=3.6, cubic=True), [1, 0, 0], vacuum=10)
surface_fcc_noisy = rattle(surface_fcc_pristine)
surface_rocksalt_pristine = surface(bulk("NaCl", "rocksalt", a=5.64, cubic=True), [1, 0, 0], vacuum=10)
surface_rocksalt_noisy = rattle(surface_rocksalt_pristine)
surface_fluorite_pristine = surface(bulk("CaF2", "fluorite", a=5.451), [1, 0, 0], vacuum=10)
surface_fluorite_noisy = rattle(surface_fluorite_pristine)
surface_1 = surface(Atoms(symbols=["O", "C", "C"], scaled_positions=[[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]], cell=[3,1,1], pbc=True), [0, 0, 1], [1, 1, 3], vacuum=10)
surface_2 = surface(Atoms(symbols=["O", "N", "N"], scaled_positions=[[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]], cell=[3,1,1], pbc=True), [0, 0, 1], [1, 1, 3], vacuum=10)
stacked_shared_species = stack(
    surface_1,
    surface_2,
    distance=1,
)
sparse = Atoms(symbols=["C"], scaled_positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
sparse *= [4, 4, 4]

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
    pytest.param(
        surface_1,
        [Cluster(range(9), dimensionality=2, classification=Classification.Surface)],
        id="stacked, part 1"
    ),
    pytest.param(
        surface_2,
        [Cluster(range(9), dimensionality=2, classification=Classification.Surface)],
        id="stacked, part 2"
    ),
    pytest.param(
        stacked_shared_species,
        [
            Cluster(range(9), dimensionality=2, classification=Classification.Surface),
            Cluster(range(9, 18), dimensionality=2, classification=Classification.Surface)
        ],
        id="stacked, shared species"
    ),
    pytest.param(
        sparse,
        [],
        id="valid region, no clusters due to sparse cell"
    ),
])
def test_clusters(system, clusters_expected):
    results = Clusterer().get_clusters(system)
    # view(system)
    # for cluster in results:
    #     indices = list(cluster.indices)
    #     if len(indices) > 1:
    #         cluster_atoms = system[indices]
    #         view(cluster_atoms)
    #         view(cluster.cell())

    # Check that correct clusters are found
    assert len(clusters_expected) == len(results)
    cluster_map = {tuple(sorted(x.indices)): x for x in results}
    for cluster_expected in clusters_expected:
        cluster = cluster_map[tuple(sorted(cluster_expected.indices))]
        assert cluster.dimensionality() == cluster_expected.dimensionality()
        assert cluster.classification() == cluster_expected.classification()