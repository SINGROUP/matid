

class SymmetryInfo():
    """A convenience class for getting symmetry information related to space
    groups.
    """
    def __init__(self, spacegroup, setting, hall_number):
        """
        Args:
            spacegroup(int): The space group number
            setting(int): The space group number does not uniquely define the
                symmetry information. You need to also define the setting. Can
                be given as a number (Same as in International Tables for
                Crystallography).
            hall_number(int): A hall number that directly specifies the space
                group and setting. Can be given instead of space group and
                setting.
        """

        # Fetch the wanted entry from a datafile
