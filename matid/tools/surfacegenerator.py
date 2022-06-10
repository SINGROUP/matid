import ase.build


class SurfaceGenerator():
    """Used to generate different kind of cuts from a given crystal cell.

    This class uses the function ase.build.surface to construct a surface from
    an arbitrary unit cell given the surface direction as Miller indices.
    Discussion about this method and the theory behind this construction
    algorithm can be found at
    https://wiki.fysik.dtu.dk/ase/ase/build/surface.html#create-specific-non-common-surfaces
    """
    def __init__(self):
        """
        Args:
            atoms(ase.Atoms): The crystal system from which the surface is
            construced.
        """

    def generate(self, atoms, miller_indices, layers, vacuum):
        """Given an arbitrary crystal lattice, transforms it to a conventional
        representation and from this conventional representation constructs a
        surface with the given Miller indices.

        Args:
            miller_indices(sequence of three ints): The surface normal
                corresponding to Miller indices (ijk).
            layers(int): The number of equivalent layers in the surface. Notice
                that this direction does not necessarily have to be orthogonal to
                the surface.
            vacuum(float): The amount of vacuum on both sides of the slab.
        """
        # Build surface with ASE
        surface = ase.build.surface(atoms, miller_indices, layers, vacuum)

        return surface
