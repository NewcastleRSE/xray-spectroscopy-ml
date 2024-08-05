"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

from ase import Atoms
from abc import ABC, abstractmethod

###############################################################################
################################## CLASSES ####################################
###############################################################################


class BaseDescriptor(ABC):
    """An abstract base class for all xanesnet descriptors."""

    @abstractmethod
    def transform(self, system: Atoms) -> np.ndarray:
        """
        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: A fingerprint feature vector for the molecular system.
        """

        pass

    @abstractmethod
    def get_nfeatures(self) -> int:
        """
        Return:
            int: Number of features for this descriptor.
        """

        pass

    @abstractmethod
    def get_type(self) -> str:
        """
        Return:
            str: descriptor type
        """

        pass
