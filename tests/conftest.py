import numpy as np
from ase import Atoms
import ase.build


def create_graphene():
	system = Atoms(
		symbols=["C", "C"],
		cell=np.array((
			[2.4595121467478055, 0.0, 0.0],
			[-1.2297560733739028, 2.13, 0.0],
			[0.0, 0.0, 20.0]
		)),
		scaled_positions=np.array((
			[1/3, 2/3, 0.5],
			[2/3, 1/3, 0.5]
		)),
		pbc=[True, True, False]
	)
	return system


def create_mos2():
	system = ase.build.mx2(
		formula='MoS2',
		kind='2H',
		a=3.18,
		thickness=3.19,
		size=(1, 1, 1),
		vacuum=0
	)
	return system

def create_si():
	system = ase.lattice.cubic.Diamond(
		size=(1, 1, 1),
		symbol='Si',
		pbc=(1, 1, 1),
		latticeconstant=5.430710
	)
	return system


def create_fe():
	system = ase.build.bulk(
		'Fe',
		crystalstructure='bcc',
		a=2.834,
	)
	return system