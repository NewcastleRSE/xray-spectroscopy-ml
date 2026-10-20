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

"""Custom analysis module for TensorBoard learning curves."""

from pathlib import Path

from xanesnet.serialization.config import Config

from ...plotters.common.style import PlotSize, get_plot_style
from ..base import CustomModule
from ..registry import CustomModuleRegistry
from .events import TrainingHistory, read_training_history
from .plotting import plot_combined_runs, plot_single_run


@CustomModuleRegistry.register("tensorboard")
class TensorBoardModule(CustomModule):
    """Plot TensorBoard learning curves for inference runs.

    Args:
        custom_module_type: Registered module name from the analysis config.
        tensorboard_dirs: TensorBoard directories aligned with inference runs.
        plot_size: Shared figure-size profile.
        latex_font: Whether to use the plotters' LaTeX-style serif font.
        y_scale: Vertical-axis scale, either ``"linear"`` or ``"log"``.
        double_width: Whether to double the base figure width.
    """

    def __init__(
        self,
        custom_module_type: str,
        tensorboard_dirs: list[str],
        plot_size: PlotSize,
        latex_font: bool,
        y_scale: str,
        double_width: bool,
    ) -> None:
        """Initialize a TensorBoard custom analysis module."""
        super().__init__(custom_module_type)
        self.tensorboard_dirs = [Path(directory) for directory in tensorboard_dirs]
        self.plot_size = plot_size
        self.latex_font = latex_font
        self.y_scale = y_scale
        self.double_width = double_width
        self.style = get_plot_style(plot_size)

    def run(self, inference_run_dirs: list[Path], prediction_names: list[str], output_dir: Path) -> None:
        """Write individual and combined learning-curve figures.

        Args:
            inference_run_dirs: Inference run directories in command-line order.
            prediction_names: Display names aligned with ``inference_run_dirs``.
            output_dir: Directory dedicated to this configured module.
        """
        runs: list[tuple[str, TrainingHistory]] = [
            (name, read_training_history(tensorboard_dir))
            for _inference_run_dir, name, tensorboard_dir in zip(
                inference_run_dirs,
                prediction_names,
                self.tensorboard_dirs,
            )
        ]

        with self.style.context(self.latex_font):
            for index, (name, history) in enumerate(runs):
                plot_single_run(
                    name,
                    history,
                    self.style,
                    output_dir / f"prediction_{index:03d}.pdf",
                    self.y_scale,
                    self.double_width,
                )
            plot_combined_runs(runs, self.style, output_dir / "combined.pdf", self.y_scale, self.double_width)

    @property
    def signature(self) -> Config:
        """Return the TensorBoard custom module signature.

        Returns:
            Configuration values needed to recreate this custom module.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "tensorboard_dirs": [str(directory) for directory in self.tensorboard_dirs],
                "plot_size": self.plot_size,
                "latex_font": self.latex_font,
                "y_scale": self.y_scale,
                "double_width": self.double_width,
            }
        )
        return signature
