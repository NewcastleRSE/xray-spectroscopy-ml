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

from .mh_cnn import MultiHead_CNN
from .mh_mlp import MultiHead_MLP
from .resolver import resolve_mh_cnn, resolve_mh_mlp  # <- triggers resolver registration

__all__ = [
    "MultiHead_MLP",
    "MultiHead_CNN",
]
