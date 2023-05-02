import pytest
from pathlib import Path

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


def assert_topology(results, expected):
    # Check that correct clusters are found
    assert len(expected) == len(results)
    cluster_map = {tuple(sorted(x.indices)): x for x in results}
    for cluster_expected in expected:
        cluster = cluster_map[tuple(sorted(cluster_expected.indices))]
        assert cluster.dimensionality() == cluster_expected.dimensionality()
        assert cluster.classification() == cluster_expected.classification()


#=========================================================================================
# Surface tests
surface_fcc = surface(bulk("Cu", "fcc", a=3.6, cubic=True), [1, 0, 0], vacuum=10)
surface_rocksalt = surface(bulk("NaCl", "rocksalt", a=5.64, cubic=True), [1, 0, 0], vacuum=10)
surface_fluorite = surface(bulk("CaF2", "fluorite", a=5.451), [1, 0, 0], vacuum=10)
@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(surface_fcc, [Cluster(range(len(surface_fcc)), dimensionality=2, classification=Classification.Surface)], id="fcc"),
    pytest.param(surface_rocksalt, [Cluster(range(len(surface_rocksalt)), dimensionality=2, classification=Classification.Surface)], id="rocksalt"),
    pytest.param(surface_fluorite, [Cluster(range(len(surface_fluorite)), dimensionality=2, classification=Classification.Surface)], id="fluorite surface"),
])
@pytest.mark.parametrize("noise", [0, 0.08])
def test_surfaces(system, noise, clusters_expected):
    system = rattle(system, noise)
    results = Clusterer().get_clusters(system)
    assert_topology(results, clusters_expected)


#=========================================================================================
# Finite tests
surface_fluorite_extended = surface_fluorite * [2, 2, 1]
@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(surface_fcc, [Cluster(range(len(surface_fcc)), dimensionality=0, classification=Classification.Class0D)], id="fcc"),
    pytest.param(surface_rocksalt, [Cluster(range(len(surface_rocksalt)), dimensionality=0, classification=Classification.Class0D)], id="rocksalt"),
    pytest.param(surface_fluorite_extended, [Cluster(range(len(surface_fluorite_extended)), dimensionality=0, classification=Classification.Class0D)], id="fluorite"),
])
@pytest.mark.parametrize("noise", [0, 0.08])
def test_finite(system, noise, clusters_expected):
    system = rattle(system, noise)
    system.set_pbc(False)
    results = Clusterer().get_clusters(system)


    assert_topology(results, clusters_expected)


#=========================================================================================
# Stacked tests
surface_1 = surface(Atoms(symbols=["O", "C", "C"], scaled_positions=[[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]], cell=[3,1,1], pbc=True), [0, 0, 1], [1, 1, 3], vacuum=10)
surface_2 = surface(Atoms(symbols=["O", "N", "N"], scaled_positions=[[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]], cell=[3,1,1], pbc=True), [0, 0, 1], [1, 1, 3], vacuum=10)
stacked_shared_species = stack(surface_1, surface_2, distance=1)
@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(
        stacked_shared_species,
        [
            Cluster(range(9), dimensionality=2, classification=Classification.Surface),
            Cluster(range(9, 18), dimensionality=2, classification=Classification.Surface)
        ],
        id="stacked, shared species"
    )
])
@pytest.mark.parametrize("noise", [0, 0.08])
def test_stacked(system, noise, clusters_expected):
    system = rattle(system, noise)
    results = Clusterer().get_clusters(system)

    # view(system)
    # for cluster in results:
    #     indices = list(cluster.indices)
    #     if len(indices) > 1:
    #         cluster_atoms = system[indices]
    #         view(cluster_atoms)
    #         view(cluster.cell())

    assert_topology(results, clusters_expected)

#=========================================================================================
# Bulk tests
bulk_one_atom = Atoms(symbols=["C"], scaled_positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)
bulk_unwrapped = Atoms(symbols=["C"], scaled_positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)
bulk_unwrapped.translate([-10, -10, -10])
@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(
        bulk_one_atom,
        [Cluster([0], dimensionality=3, classification=Classification.Class3D)],
        id="bulk, only one atom in cluster, still a valid cluster"
    ),
    pytest.param(
        bulk_unwrapped,
        [Cluster([0], dimensionality=3, classification=Classification.Class3D)],
        id="bulk, unwrapped coordinates"
    )
])
@pytest.mark.parametrize("noise", [0])
def test_bulk(system, noise, clusters_expected):
    system = rattle(system, noise)
    results = Clusterer().get_clusters(system)
    assert_topology(results, clusters_expected)


#=========================================================================================
# Misc tests
broken = Atoms(
    symbols=["H", "O", "O", "O", "O", "O", "O", "O"],
    scaled_positions=[
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0.5, 0.5, 0],
        [0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ],
    cell=[2, 2, 2],
    pbc=True
)

broken *= [1, 1, 3]
del broken[1:8]
ase.build.add_vacuum(broken, 10)

sparse = Atoms(symbols=["C"], scaled_positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
@pytest.mark.parametrize("system, clusters_expected", [
    pytest.param(sparse, [], id="valid region, no clusters due to sparse cell"),
    pytest.param(broken, [
        Cluster(range(1, 17), dimensionality=2, classification=Classification.Surface)
    ], id="remove unconnected outliers from region"),
])
def test_misc(system, clusters_expected):
    results = Clusterer().get_clusters(system)
    assert_topology(results, clusters_expected)