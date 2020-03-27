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
            space_group (int): The space group number for this set.
            representative (str): Algebraic expression that represents this set: all
                other positions are generated from symmetry.
            multiplicity (int): The multiplicity of the set: how many atoms are
                in this set.
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

    def __repr__(self): 
        return "Wyckoff letter: {}, element: {}, multiplicity: {}, space group: {}".format(
            self.wyckoff_letter,
            self.element,
            self.multiplicity,
            self.space_group
        )
  
    def __str__(self): 
        return self.__repr__()

    def __eq__(self, other):
        if self.wyckoff_letter != other.wyckoff_letter: return False
        if self.atomic_number != other.atomic_number: return False
        if self.element != other.element: return False
        if self.space_group != other.space_group: return False
        if self.multiplicity != other.multiplicity: return False
        if self.x != other.x: return False
        if self.y != other.y: return False
        if self.z != other.z: return False

        return True
