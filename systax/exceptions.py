class SystaxError(Exception):
    pass


class ClassificationError(SystaxError):
    """Indicates that there was an error in finding a surface.
    """
    pass


class CellNormalizationError(SystaxError):
    """For errors in finding the normalized cell.
    """
    pass
