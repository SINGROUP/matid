from __future__ import absolute_import, division, print_function, unicode_literals


class SystaxError(Exception):
    def __init__(self, message, value=None):
        self.value = value
        Exception.__init__(self, message)


class ClassificationError(SystaxError):
    """Indicates that there was an error in finding a surface.
    """
    pass


class CellNormalizationError(SystaxError):
    """For errors in finding the normalized cell.
    """
    pass
