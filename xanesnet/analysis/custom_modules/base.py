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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Base interface for custom analysis modules."""

from abc import ABC, abstractmethod
from pathlib import Path

from xanesnet.serialization.config import Config


class CustomModule(ABC):
    """Base class for a fully custom analysis modules.

    Args:
        custom_module_type: Registered module name from the analysis configuration.
    """

    def __init__(self, custom_module_type: str) -> None:
        """Initialize a custom module instance."""
        self.custom_module_type = custom_module_type

    @abstractmethod
    def run(
        self,
        inference_run_dirs: list[Path],
        prediction_names: list[str],
        output_dir: Path,
    ) -> None:
        """Run the custom analysis module.

        Args:
            inference_run_dirs: Inference run directories in command-line order.
            prediction_names: Display names aligned with ``inference_run_dirs``.
            output_dir: Directory dedicated to this configured module.
        """
        ...

    @property
    def signature(self) -> Config:
        """Return the custom module signature.

        Returns:
            Configuration values needed to recreate this module.
        """
        return Config({"custom_module_type": self.custom_module_type})

    def __str__(self) -> str:
        """Return the short display label of this custom module.

        Returns:
            The registered module name.
        """
        return self.custom_module_type

    def __repr__(self) -> str:
        """Return a detailed representation of this custom module.

        Returns:
            The class name and signature configuration.
        """
        args = ", ".join(f"{key}={value!r}" for key, value in self.signature.as_dict().items())
        return f"{type(self).__name__}({args})"
