import pytest
import ase.io
import ase.build
from ase.visualize import view

from matid.symmetry.wyckoffset import WyckoffSet
from matid.symmetry.symmetryanalyzer import SymmetryAnalyzer
from conftest import create_mos2, create_graphene, create_si, create_fe, create_sic


mos2 = create_mos2()
mos2_rotated = mos2.copy()
cell = mos2.get_cell()
rotation_axis = cell[0] + cell[1]
mos2_rotated.rotate(180, rotation_axis, center=mos2[0].position)
mos2_supercell = mos2.copy()
mos2_supercell *= [5, 5, 1]
mos2_vacuum = ase.build.mx2(
    formula="MoS2",
    kind="2H",
    a=3.18,
    thickness=3.19,
    size=(1, 1, 1),
    vacuum=8
)
graphene = create_graphene()
graphene_rotated = graphene.copy()
cell = graphene.get_cell()
rotation_axis = cell[0] + cell[1]
graphene_rotated.rotate(180, rotation_axis, center=graphene[0].position)


material_id_mos2 = "7ZWsaQmykAhJ4SK8Fkp-lod63p5m"
wyckoff_sets_mos2 = [
    WyckoffSet(element="Mo", wyckoff_letter="a"),
    WyckoffSet(element="S", wyckoff_letter="h"),
]
material_id_graphene = "jPcWsq-Rb0gtgx2krJDmLvQUxpcL"
wyckoff_sets_graphene = [
    WyckoffSet(element="C", wyckoff_letter="c"),
]
material_id_si = "fh3UBjhUVm4nxzeRd2JJuqw5oXYa"
wyckoff_sets_si = [
    WyckoffSet(element="Si", wyckoff_letter="a"),
]
material_id_fe = "TpsQ0uU5Y6ureJqxP0lQqPVGBh13"
wyckoff_sets_fe = [
    WyckoffSet(element="Fe", wyckoff_letter="a"),
]
material_id_sic = "kbzX8WFgkBEIQD0kAYJwgril4ck8"
wyckoff_sets_sic = [
    WyckoffSet(element="C", wyckoff_letter="a"),
    WyckoffSet(element="Si", wyckoff_letter="c"),
]
@pytest.mark.parametrize("system, material_id_expected, wyckoff_sets_expected", [
    pytest.param(mos2, material_id_mos2, wyckoff_sets_mos2, id="2D, non-flat."),
    pytest.param(mos2_rotated, material_id_mos2, wyckoff_sets_mos2, id="2D, non-flat, rotated"),
    pytest.param(mos2_vacuum, material_id_mos2, wyckoff_sets_mos2, id="2D, non-flat, with vacuum"),
    pytest.param(mos2_supercell, material_id_mos2, wyckoff_sets_mos2, id="2D, non-flat, supercell"),
    pytest.param(graphene, material_id_graphene, wyckoff_sets_graphene, id="2D, flat."),
    pytest.param(graphene_rotated, material_id_graphene, wyckoff_sets_graphene, id="2D, flat, rotated"),
    pytest.param(create_si(cubic=True), material_id_si, wyckoff_sets_si, id="3D, cubic, silicon diamond"),
    pytest.param(create_si(cubic=False), material_id_si, wyckoff_sets_si, id="3D, primitive, silicon diamond"),
    pytest.param(create_fe(cubic=True), material_id_fe, wyckoff_sets_fe, id="3D, cubic, iron BCC"),
    pytest.param(create_fe(cubic=False), material_id_fe, wyckoff_sets_fe, id="3D, primitive, iron BCC"),
    pytest.param(create_sic(cubic=True), material_id_sic, wyckoff_sets_sic, id="3D, cubic, SiC zinc blende"),
    pytest.param(create_sic(cubic=False), material_id_sic, wyckoff_sets_sic, id="3D, primitive, SiC zinc blende"),
])
def test_conventional_system(system, material_id_expected, wyckoff_sets_expected):
    analyzer = SymmetryAnalyzer(system)
    material_id = analyzer.get_material_id()
    wyckoff_sets_conv = analyzer.get_wyckoff_sets_conventional()
    conv_sys = analyzer.get_conventional_system()
    assert_wyckoff_sets(wyckoff_sets_conv, wyckoff_sets_expected)
    assert material_id == material_id_expected


def assert_wyckoff_sets(a, b):
    '''Checks that the given Wyckoff sets are the same.'''
    assert len(a) == len(b)

    def sets(wyckoff_sets):
        sets = set()
        for wyckoff_set in wyckoff_sets:
            sets.add((wyckoff_set.element, wyckoff_set.wyckoff_letter))
        return sets

    assert sets(a) == sets(b)