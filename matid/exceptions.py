class MatIDError(Exception):
    def __init__(self, message, value=None):
        self.value = value
        Exception.__init__(self, message)


class ClassificationError(MatIDError):
    """Indicates that there was an error in finding a surface.
    """
    pass


class CellNormalizationError(MatIDError):
    """For errors in finding the normalized cell.
    """
    pass
