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

from ase import Atoms

from xanesnet.descriptor.base_descriptor import BaseDescriptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class MACE(BaseDescriptor):
    """
    A class for reading the descriptor straight from a file. It tries to avoid
    doing any of the fancy stuff the other descriptors do. Only reads the file
    """

    def transform(self, system: Atoms) -> int:
        return 0

    def get_nfeatures(self) -> int:
        return 0

    def get_type(self) -> str:
        return "mace"
