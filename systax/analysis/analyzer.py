from __future__ import absolute_import, division, print_function

import sys

import abc
from abc import abstractmethod

from systax.data.constants import SPGLIB_PRECISION

__metaclass__ = type

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Analyzer(ABC):
    """A base class for all classes that are used to analyze structures.
    """
    def __init__(self, system=None, spglib_precision=None, vacuum_gaps=None, unitcollection=None, unit_cell=None):
        """
        Args:
            system (ASE.Atoms): The system to inspect.
            spglib_precision (float): The tolerance for the symmetry detection
                done by spglib.
        """
        self.system = system
        self.vacuum_gaps = vacuum_gaps
        self.unitcollection = unitcollection
        self.unit_cell = unit_cell
        if spglib_precision is None:
            self.spglib_precision = SPGLIB_PRECISION
        else:
            self.spglib_precision = spglib_precision

        self.reset()

    def set_system(self, system):
        """Sets a new system for analysis.
        """
        self.reset()
        self.system = system

    @abstractmethod
    def reset(self):
        """Used to reset all the cached values.
        """
