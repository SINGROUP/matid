class Distances():
    """Container for all distance information that has been extracted from a
    system.
    """
    def __init__(self, disp_tensor_mic, disp_factors, disp_tensor_finite, dist_matrix_mic, dist_matrix_radii_mic):
        self.disp_tensor_mic = disp_tensor_mic
        self.disp_factors = disp_factors
        self.disp_tensor_finite = disp_tensor_finite
        self.dist_matrix_mic = dist_matrix_mic
        self.dist_matrix_radii_mic = dist_matrix_radii_mic
