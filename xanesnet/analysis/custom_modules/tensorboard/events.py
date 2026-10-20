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

"""Read training metrics from TensorBoard event files."""

from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class ScalarSeries:
    """Scalar values indexed by one-based epoch.

    Args:
        epochs: One-based epoch for each scalar value.
        values: Scalar value recorded at each epoch.
    """

    epochs: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class TrainingHistory:
    """Training metrics read from one TensorBoard directory.

    Args:
        training_loss: Training loss scalar series.
        validation_loss: Validation loss scalar series.
        learning_rate: Learning-rate scalar series.
    """

    training_loss: ScalarSeries
    validation_loss: ScalarSeries
    learning_rate: ScalarSeries


def _read_scalar_series(accumulator: EventAccumulator, tag: str) -> ScalarSeries:
    """Read one TensorBoard scalar tag in ascending epoch order.

    Args:
        accumulator: Reloaded TensorBoard event accumulator.
        tag: Scalar tag to read.

    Returns:
        Scalar values and their one-based epoch indices.
    """
    events = sorted(accumulator.Scalars(tag), key=lambda event: event.step)
    return ScalarSeries(
        epochs=tuple(event.step + 1 for event in events),
        values=tuple(event.value for event in events),
    )


def read_training_history(tensorboard_dir: Path) -> TrainingHistory:
    """Read XANESNET training, validation, and learning-rate scalars.

    Args:
        tensorboard_dir: Directory containing TensorBoard event files.

    Returns:
        Training history for the directory.
    """
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return TrainingHistory(
        training_loss=_read_scalar_series(accumulator, "loss/train"),
        validation_loss=_read_scalar_series(accumulator, "loss/valid"),
        learning_rate=_read_scalar_series(accumulator, "other/learning_rate"),
    )


def minimum_validation_point(history: TrainingHistory) -> tuple[int, float]:
    """Return the epoch and value of the minimum validation loss.

    Args:
        history: Training history to inspect.

    Returns:
        One-based epoch and minimum validation loss.
    """
    return min(
        zip(history.validation_loss.epochs, history.validation_loss.values),
        key=lambda point: point[1],
    )
