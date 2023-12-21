"""
XANESNET

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

from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

from xanesnet.descriptor.vector_descriptor import VectorDescriptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class DIRECT(VectorDescriptor):
    """
    A class for reading the descriptor straight from a file. It tries to avoid
    doing any of the fancy stuff the other descriptors do. Only reads the file
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 8.0,
        use_charge=False,
        use_spin=False,
    ):
        """
        Args:
        """
        super().__init__(r_min, r_max, use_charge, use_spin)

    def transform(self,) -> int:
        return 0

    def get_number_of_features(self) -> int:
        return 0

    def process(self,) -> int:
        return 0
