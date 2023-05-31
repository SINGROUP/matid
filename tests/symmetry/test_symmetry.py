import pytest
import numpy as np
import ase.io
from ase import Atoms
import ase.build
from ase.visualize import view

from matid.symmetry.symmetryanalyzer import SymmetryAnalyzer
from conftest import create_mos2, create_graphene


mos2 = create_mos2()
mos2_rotated = mos2.copy()
cell = mos2.get_cell()
rotation_axis = cell[0] + cell[1]
mos2_rotated.rotate(180, rotation_axis, center=mos2[0].position)
graphene = create_graphene()
graphene_rotated = graphene.copy()
cell = graphene.get_cell()
rotation_axis = cell[0] + cell[1]
graphene_rotated.rotate(180, rotation_axis, center=graphene[0].position)


@pytest.mark.parametrize("system, material_id_expected", [
    # pytest.param(mos2, "7ZWsaQmykAhJ4SK8Fkp-lod63p5m", id="2D, non-flat."),
    pytest.param(mos2_rotated, "7ZWsaQmykAhJ4SK8Fkp-lod63p5m", id="2D, non-flat, rotated"),
    pytest.param(graphene, "jPcWsq-Rb0gtgx2krJDmLvQUxpcL", id="2D, flat."),
    pytest.param(graphene_rotated, "jPcWsq-Rb0gtgx2krJDmLvQUxpcL", id="2D, flat, rotated"),
])
def test_material_id(system, material_id_expected):
	analyzer = SymmetryAnalyzer(system)
	material_id = analyzer.get_material_id()
	assert material_id == material_id_expected