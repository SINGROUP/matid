from systax.analysis.analyzer import Analyzer


class SurfaceAnalyzer(Analyzer):
    """Used to analyze properties of surfaces.
    """

    def __init__(self, atoms=None, component=None):
        """
        """
        super(SurfaceAnalyzer, self).__init__(atoms, spglib_precision=None, vacuum_gaps=None)

    def reset(self):
        """
        """

    def get_miller_indices(self):
        """Detects the Miller indices (hjk) of the surface with respect to the
        conventional representation of the cell.
        """
        surface_atoms = atoms
        bulk_analyzer = self.component.bulk_analyzer
        origin_shift = bulk_analyzer._get_spglib_origin_shift()
        transformation_matrix = bulk_analyzer._get_spglib_transformation_matrix()
        symmetry_dataset = self.component.bulk_analyzer.get_symmetry_dataset()

    def get_thickness(self):
        """
        """

    def get_top_indices(self):
        """Get the indices of the top surfaces. The top surface is determined
        by:
            vector a is the first
        """
