from ase.build import fcc111, fcc100, bcc100, fcc110
from ase.visualize import view

from sklearn.decomposition import PCA

import numpy as np

import spglib

# system = bcc100('Al', size=(8, 8, 4), vacuum=8, a=3.5)
system = fcc110('Al', size=(8, 8, 4), vacuum=8, a=4.5)
# system = fcc100('Al', size=(8, 8, 4), vacuum=8, a=4.5)
# system = system.repeat([2, 2, 2])
# view(system_bcc)

positions = system.get_positions()
numbers = system.get_atomic_numbers()
cell = system.get_cell()

# Get principal direction with PCA
pca = PCA(svd_solver="full")
pca.fit(positions)
components = pca.components_
# print(components)

# The smallest component is the orhogonal one
z_direction = components[-1, :]
# print(z_direction)

# Find out the cell direction that corresponds to the orthogonal one
dots = np.abs(np.dot(z_direction, cell.T))
z_cell_vector_index = np.argmax(dots)
z_dir_ideal = np.array([0, 0, 0], dtype=float)
z_dir_ideal[z_cell_vector_index] = 0.145

# Pick a seed point from the middle of the cell
middle = np.mean(positions, axis=0)
distances = np.linalg.norm(positions - middle)
seed_index = np.argmin(distances)
seed_atom = system[seed_index]
seed_pos = np.array(seed_atom.position)
# print(seed_atom)

# Calculate the pairwise displacement vectors: the displacement tensor
pos = np.array(system.get_positions())
disp_tensor = pos[:, None, :] - pos[None, :, :]
displacement_tensor = disp_tensor

# Get the vectors that span from the seed to all other atoms
spans = displacement_tensor[:, seed_index]
spans = spans[np.arange(len(spans)) != seed_index]
# print(spans)

# Find the spans that when doubled lead to another identical atom within
# some tolerance
tolerance = 0.1
test_vectors = np.array(seed_pos) + 2*spans
pos_test = pos[np.arange(len(pos)) != seed_index]

disp = pos_test[:, None, :] - test_vectors[None, :, :]
distances = np.linalg.norm(disp, axis=2)
_, span_indices = np.where(distances < tolerance)
spans = spans[span_indices]
# print(spans)

# Select maximally orthogonal directions for next phase. Algorithm from
# https://stackoverflow.com/questions/5721523/how-to-get-the-maximally-independent-vectors-given-a-set-of-vectors-in-matlab
i_considered = 0
q, r = np.linalg.qr(spans.T)
rInd = []
for j in range(0, r.shape[1]):
    if r[i_considered, j] != 0:
        rInd.append(r[:, j])
        i_considered = i_considered + 1
    if i_considered >= r.shape[0]:
        break

basis = np.array(rInd)
print(basis)

# Group spans to the three direction that were found
basis_lenghts = np.linalg.norm(basis, axis=1)
span_lenghts = np.linalg.norm(spans, axis=1)

angle_tol = 2
angles = np.zeros((3, len(spans)))
for i_bas, bas in enumerate(basis):
    basis_length = basis_lenghts[i_bas]
    dot = np.dot(bas, spans.T)
    cos = dot/(span_lenghts*basis_length)
    cos = np.clip(cos, -1.0, 1.0)
    i_angles = np.arccos(cos)
    angles[i_bas, :] = i_angles*180/np.pi

angle_mask = np.where(angles < angle_tol)
# print(angle_mask)

groups = []
for i_basis in range(3):
    basis_mask = np.where(angle_mask[0] == i_basis)
    span_mask = angle_mask[1][basis_mask]
    span_group = spans[span_mask]
    groups.append(span_group)

# Choose the smallest found basis from each group as a first try
new_basis = np.zeros([3, 3])
for i_group, group in enumerate(groups):
    norms = np.linalg.norm(group, axis=1)
    i_min_basis = np.argmin(norms)
    min_basis = group[i_min_basis]
    new_basis[i_group, :] = min_basis
# print(new_basis)

# Find the atoms within the found cell
new_basis_inverse = np.linalg.inv(new_basis.T)
vec_new = np.dot(pos - seed_pos, new_basis_inverse.T)

cell_pos = []
cell_numbers = []
precision = 1E-5
for i_pos, pos in enumerate(vec_new):
    # print(pos)
    x = 0 <= pos[0] < 1 - precision
    y = 0 <= pos[1] < 1 - precision
    z = 0 <= pos[2] < 1 - precision
    # print(z)
    if x and y and z:
        cell_pos.append(pos)
        cell_numbers.append(numbers[i_pos])

cell_pos = np.array(cell_pos)
# print(cell_pos)

# Run spglib on the found cell to get the conventional cell
material = (new_basis, cell_pos, cell_numbers)
dataset = spglib.get_symmetry_dataset(material)
space_group = dataset["number"]
std_lattice = dataset["std_lattice"]
# print(space_group)
# print(std_lattice)

# Find the surface Miller indices from the transformation that is defined
# between the normalized cell and the input cell
transformation = dataset["transformation_matrix"]
shift = dataset["origin_shift"]
z_new_basis = np.dot(z_dir_ideal, new_basis_inverse.T)
z_norm_std = np.dot(transformation, z_new_basis)
# print(z_norm_std)

# Find the reciprocal cell vectors for the standardized cell, then determine
# the direction in this basis.
reciprocal_lattice = np.linalg.inv(std_lattice.T)
reciprocal_lattice_inv = np.linalg.inv(reciprocal_lattice.T)
miller_coords = np.dot(z_norm_std, reciprocal_lattice_inv.T)
nonzero_ind = np.where(miller_coords != 0 & (abs(miller_coords) < 1))
nonzeros = miller_coords[nonzero_ind]
min_nonzero = np.min(nonzeros)
if min_nonzero < 1:
    miller_coords /= min_nonzero
rounded_miller = np.round(miller_coords).astype(dtype=int)
print(rounded_miller)
