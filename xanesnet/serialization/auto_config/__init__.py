# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Automatic model and encoding configuration resolution from prepared datasets.

Model resolvers live in each model subpackage (``models/<name>/resolver.py``)
and are imported by the model subpackage ``__init__.py``, which triggers their
registration with :data:`~xanesnet.serialization.auto_config.registries.ModelAutoResolver`
as a side effect.  Encoding resolvers are defined in each encoding module
(``encodings/<name>.py``) and are registered with
:data:`~xanesnet.serialization.auto_config.registries.EncodingAutoResolver`
when the encoding module is imported.
"""

from .core import resolve_auto_encoding_config, resolve_auto_model_config

__all__ = ["resolve_auto_encoding_config", "resolve_auto_model_config"]
