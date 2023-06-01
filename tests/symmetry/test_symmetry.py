import pytest
import numpy as np
import ase.io
from ase import Atoms
import ase.build
from ase.visualize import view

from matid.symmetry.symmetryanalyzer import SymmetryAnalyzer
from conftest import create_mos2, create_graphene, create_si, create_fe


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


@pytest.mark.parametrize("system, material_id_expected", [
    pytest.param(mos2, "7ZWsaQmykAhJ4SK8Fkp-lod63p5m", id="2D, non-flat."),
    pytest.param(mos2_rotated, "7ZWsaQmykAhJ4SK8Fkp-lod63p5m", id="2D, non-flat, rotated"),
    pytest.param(mos2_vacuum, "7ZWsaQmykAhJ4SK8Fkp-lod63p5m", id="2D, non-flat, with vacuum"),
    pytest.param(mos2_supercell, "7ZWsaQmykAhJ4SK8Fkp-lod63p5m", id="2D, non-flat, supercell"),
    pytest.param(graphene, "jPcWsq-Rb0gtgx2krJDmLvQUxpcL", id="2D, flat."),
    pytest.param(graphene_rotated, "jPcWsq-Rb0gtgx2krJDmLvQUxpcL", id="2D, flat, rotated"),
    pytest.param(create_si(), "fh3UBjhUVm4nxzeRd2JJuqw5oXYa", id="3D, silicon diamond"),
    pytest.param(create_fe(), "TpsQ0uU5Y6ureJqxP0lQqPVGBh13", id="3D, iron BCC"),
])
def test_material_id(system, material_id_expected):
	analyzer = SymmetryAnalyzer(system)
	material_id = analyzer.get_material_id()
	assert material_id == material_id_expected
