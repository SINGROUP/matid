class WyckoffSet():
    """Represents a group of atoms in a certain Wyckoff position, for a certain
    space group.
    """
    def __init__(
            self,
            wyckoff_letter=None,
            atomic_number=None,
            element=None,
            indices=None,
            x=None,
            y=None,
            z=None,
            space_group=None,
            representative=None,
            multiplicity=None,
            ):
        """
        Args:
            wyckoff_letter (str): The letter for the Wyckoff position.
            atomic_number (int): Atomic number stored in this Wyckoff position.
            element (str): The chemical symbol of the element.
            indices (list): Indices corresponding to atoms in this Wyckoff
                group in a structure.
            x (float): The free parameter for the a-vector.
            y (float): The free parameter for the b-vector.
            z (float): The free parameter for the c-vector.
        """
        self.wyckoff_letter = wyckoff_letter
        self.atomic_number = atomic_number
        self.element = element
        self.space_group = space_group
        self.indices = indices
        self.multiplicity = multiplicity
        self.representative = representative
        self.x = x
        self.y = y
        self.z = z
