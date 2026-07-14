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

"""Factor-scaling spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError

from .affine import AffineEncoding, build_parameter_rows
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("scale")
class ScaleEncoding(AffineEncoding):
    """Factor-scaling encoding.

    Divides spectra by a ``factor`` during encoding (``x / factor``) and
    multiplies by the same factor during decoding (``x * factor``), without any
    centering. Typically the factor is the standard deviation of the training
    spectra, bringing target magnitudes into a numerically convenient range. It
    can be provided explicitly or resolved automatically from the training data
    via the ``"auto"`` token.

    Two orthogonal switches control how the factor is shaped:

    * ``per_point`` selects whether the factor varies across the spectrum
      (a length-``N`` vector) or is a single global scalar shared across all
      points.
    * ``per_element`` selects whether a separate factor is used for each
      absorbing element (chosen per sample from its atomic number) or a single
      factor is shared across all elements.

    When ``per_element`` is false, ``factor`` is a flat list. When it is true,
    it is a list of rows, one row per entry of ``elements``.

    Zero or near-zero factors (below ``1e-12``) are silently replaced with
    ``1.0``, which is equivalent to skipping scaling at those points. This
    handles flat spectral regions (e.g. the pre-edge) where the per-point
    standard deviation vanishes.

    Args:
        encoding_type: Identifier string for this encoding type.
        factor: Divisor applied during encoding. A flat list when
            ``per_element`` is false, or one row per element (aligned with
            ``elements``) when true.
        per_point: Whether scaling is applied per spectrum point (``True``) or
            globally with a single shared factor (``False``).
        per_element: Whether the factor is selected per absorbing element
            (``True``) or shared across all elements (``False``).
        elements: Atomic numbers aligned row-wise with ``factor`` when
            ``per_element`` is true; ignored otherwise.

    Raises:
        ConfigError: If ``factor`` is empty.
    """

    def __init__(
        self,
        encoding_type: str,
        factor: list[float] | list[list[float]],
        per_point: bool,
        per_element: bool = False,
        elements: list[int] | None = None,
    ) -> None:
        """Initialize ``ScaleEncoding``."""
        if per_element:
            factor_tensor = build_parameter_rows(encoding_type, "factor", factor)  # type: ignore[arg-type]
        else:
            if not factor:
                raise ConfigError(f"{encoding_type} requires a non-empty factor.")
            factor_tensor = torch.tensor(factor, dtype=torch.float32)

        # Replace near-zero factors with 1.0 (identity scaling) so that flat
        # spectral regions (e.g. pre-edge where std \approx 0) do not cause
        # division-by-zero during encoding.
        factor_tensor = torch.where(
            factor_tensor < 1e-12,
            torch.ones_like(factor_tensor),
            factor_tensor,
        )

        self.per_point = per_point
        super().__init__(encoding_type, torch.zeros_like(factor_tensor), factor_tensor, per_element, elements)

    @property
    def signature(self) -> list[Config]:
        """Return the scale-encoding signature.

        Returns:
            Configuration values needed to recreate this encoding.
        """
        sig = super().signature[0]
        values: dict[str, object] = {
            "factor": self.scale.tolist(),
            "per_point": self.per_point,
            "per_element": self.per_element,
        }
        if self.per_element:
            values["elements"] = list(self.elements or [])
        sig.update_with_dict(values)
        return [sig]
