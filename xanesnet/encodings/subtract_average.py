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

"""Average-spectrum subtraction spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError

from .affine import AffineEncoding, build_parameter_rows
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("subtract_average")
class SubtractAverageEncoding(AffineEncoding):
    """Average-spectrum subtraction encoding.

    Centers spectra during encoding by subtracting a fixed average spectrum
    (``x - average``) and restores them during decoding by adding it back
    (``x + average``). ``average`` is a per-point vector with one entry per
    spectrum point, typically the mean spectrum of the training data. It can be
    provided explicitly or resolved automatically from the training data via the
    ``"auto"`` token. Training then operates on residuals about the mean
    spectrum.

    The ``per_element`` switch selects whether a separate average spectrum is
    used for each absorbing element (chosen per sample from its atomic number)
    or a single average is shared across all elements. When ``per_element`` is
    false, ``average`` is a flat list; when true, it is a list of rows, one row
    per entry of ``elements``.

    Args:
        encoding_type: Identifier string for this encoding type.
        average: Per-point average spectrum. A flat list when ``per_element`` is
            false, or one row per element (aligned with ``elements``) when true.
        per_element: Whether the average is selected per absorbing element
            (``True``) or shared across all elements (``False``).
        elements: Atomic numbers aligned row-wise with ``average`` when
            ``per_element`` is true; ignored otherwise.

    Raises:
        ConfigError: If ``average`` is empty.
    """

    def __init__(
        self,
        encoding_type: str,
        average: list[float] | list[list[float]],
        per_element: bool = False,
        elements: list[int] | None = None,
    ) -> None:
        """Initialize ``SubtractAverageEncoding``."""
        if per_element:
            shift = build_parameter_rows(encoding_type, "average", average)  # type: ignore[arg-type]
        else:
            if not average:
                raise ConfigError(f"{encoding_type} requires a non-empty average.")
            shift = torch.tensor(average, dtype=torch.float32)

        super().__init__(encoding_type, shift, torch.ones_like(shift), per_element, elements)

    @property
    def signature(self) -> list[Config]:
        """Return the subtract-average-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        values: dict[str, object] = {
            "average": self.shift.tolist(),
            "per_element": self.per_element,
        }
        if self.per_element:
            values["elements"] = list(self.elements or [])
        sig.update_with_dict(values)
        return [sig]
