import os
import json

import ase.io

from matid import Classifier
from matid import SymmetryAnalyzer
from matid.classifications import Class3D, Material2D, Surface

# This is a folder containing 10 different extended XYZ files.
inpath = "./structures"

# Lets find all XYZ files and read the geometries as ASE.Atoms objects in to a
# list
geometries = []
for root, dirs, files in os.walk(inpath):
    for i_file in files:
        # if i_file.endswith("C96Si96+C54H16.xyz"):
        if i_file.endswith("xyz"):
            i_atoms = ase.io.read("{}/{}".format(root, i_file))
            # view(i_atoms)
            geometries.append((i_file, i_atoms))

# Create a Classifier instance. The default settings are used here
classifier = Classifier()

# Get a classification result for each geometry
classifications = []
for i_file, i_geom in geometries:
    print("Classifying")
    i_cls = classifier.classify(i_geom)
    print("Done")
    classifications.append(i_cls)

# Create a summary of the geometries
summary = {}
for (i_file, i_geom), i_cls in zip(geometries, classifications):
    i_type = type(i_cls)
    i_atoms = i_cls.atoms

    i_data = {
        "system_type": str(i_cls),
    }

    # Get symmetry information
    blk_cell = None
    if i_type == Class3D:
        blk_cell = i_atoms
    elif i_type == Surface:
        blk_cell = i_cls.prototype_cell
    if blk_cell is not None:
        symm_analyzer = SymmetryAnalyzer(blk_cell)
        formula = i_atoms.get_chemical_formula()
        crystal_system = symm_analyzer.get_crystal_system()
        bravais_lattice = symm_analyzer.get_bravais_lattice()
        space_group = symm_analyzer.get_space_group_number()
        i_data["space_group_number"] = space_group
        i_data["crystal_system"] = crystal_system
        i_data["bravais_lattice"] = bravais_lattice

    # Get the outlier information from two-dimensional systems
    if i_type == Surface or i_type == Material2D:
        outlier_indices = i_cls.outliers
        outlier_formula = i_atoms[outlier_indices].get_chemical_formula()
        i_data["outlier_indices"] = outlier_indices.tolist()
        i_data["outlier_formula"] = outlier_formula

    summary[i_file] = i_data

# Write a summary of the results
with open("summary.json", "w") as fout:
    fout.write(json.dumps(summary, indent=2, sort_keys=True))
