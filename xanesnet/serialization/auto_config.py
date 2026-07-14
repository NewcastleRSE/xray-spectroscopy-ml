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

"""Automatic model-configuration resolution from prepared datasets."""

import logging
from collections.abc import Callable
from typing import Any

import torch

from xanesnet.batchprocessors import BatchProcessor, BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.utils.exceptions import ConfigError

from .config import Config

AUTO_VALUE = "auto"
_INVERSE_DATASET_SUFFIX = "_inverse"
AutoResolver = Callable[[dict[str, Any], Any], dict[str, Any]]


def _is_inverse_dataset(dataset_type: str) -> bool:
    """Return ``True`` if ``dataset_type`` identifies an inverse prediction dataset.

    Args:
        dataset_type: Registered dataset type name.

    Returns:
        ``True`` when the dataset type ends with ``"_inverse"``.
    """
    # TODO This is not optimal! Will miss for example _mp!
    return dataset_type.endswith(_INVERSE_DATASET_SUFFIX)


def resolve_auto_encoding_config(config: Config, dataset: Dataset) -> Config:
    """Resolve encoding fields set to ``"auto"`` from a prepared dataset.

    The input config is expected to have passed schema-backed training
    validation. This function performs only the dataset-dependent finalization
    for the top-level ``encodings`` list: it accumulates overall and
    per-absorbing-element statistics of the training targets in a single
    streaming pass (so the full target matrix is never materialized), asks the
    encoding-specific resolver for concrete values (e.g. per-point statistics
    over the training spectra), and returns a new ``Config`` without mutating
    the input config.

    .. note::

        For inverse prediction datasets the targets are structural
        descriptors or properties, not spectra. Auto-resolution of encoding
        fields is skipped for inverse datasets; use an explicit
        ``encoding_type: identity`` or manually specify encoding parameters.

    Args:
        config: Validated training configuration.
        dataset: Prepared training dataset used to derive automatic encoding
            values.

    Returns:
        New configuration with requested automatic encoding fields replaced by
        concrete values.

    Raises:
        ConfigError: If automatic fields are requested for an encoding that has
            no resolver or if the resolver does not produce a requested field.
    """
    config_raw = config.as_dict()
    encodings = config_raw.get("encodings", [])
    model_type = config_raw["model"]["model_type"]

    if not any(_requested_auto_fields(item) for item in encodings):
        return Config(config_raw)

    if _is_inverse_dataset(dataset.dataset_type):
        logging.warning(
            "Skipping auto-encoding resolution for inverse dataset '%s'. "
            "Encoding fields set to 'auto' in an inverse config must be "
            "resolved manually or use encoding_type: identity.",
            dataset.dataset_type,
        )
        return Config(config_raw)

    batchprocessor = BatchProcessorRegistry.create((dataset.dataset_type, model_type))
    statistics = _collect_target_statistics(dataset, batchprocessor)

    for item in encodings:
        auto_fields = _requested_auto_fields(item)
        if not auto_fields:
            continue

        encoding_type = item["encoding_type"]
        resolver = _encoding_resolver_for(encoding_type)
        resolved_fields = resolver(item, statistics)

        for field in sorted(auto_fields):
            if field not in resolved_fields:
                raise ConfigError(f"Automatic resolver for encoding '{encoding_type}' did not produce {field}.")

            item[field] = resolved_fields[field]
            logging.info(
                "Resolved encoding '%s' field %s=%s from dataset '%s' with %s.",
                encoding_type,
                field,
                _format_resolved_value(resolved_fields[field]),
                dataset.dataset_type,
                type(batchprocessor).__name__,
            )

    return Config(config_raw)


def resolve_auto_model_config(config: Config, dataset: Dataset, encoding: SpectraEncoding) -> Config:
    """Resolve model fields set to ``"auto"`` from a prepared dataset.

    The input config is expected to have passed schema-backed training
    validation, including validation that ``"auto"`` is used only for supported
    top-level model fields. This function only performs the dataset-dependent
    finalization: it uses the registered batch processor for the dataset/model
    pair to prepare one sample, maps the raw input and target into the spaces
    the model consumes and predicts via the batch processor's
    :meth:`~xanesnet.batchprocessors.base.BatchProcessor.encode_input` and
    :meth:`~xanesnet.batchprocessors.base.BatchProcessor.encode_target`, asks
    the model-specific resolver for concrete dimensions, and returns a new
    ``Config`` without mutating the input config.

    The batch processor is created *with* the resolved encoding so that
    resolved input and output dimensions match the tensors the model actually
    receives at train time: for forward datasets the encoding shapes the target
    (and thus the output size), while for inverse datasets the encoding shapes
    the spectral input (and thus the input size).

    Args:
        config: Validated training configuration.
        dataset: Prepared training dataset used to derive automatic model
            values.
        encoding: Spectra encoding forwarded to the batch processor so that
            resolved dimensions match the model's encoded prediction space
            (forward) or encoded input space (inverse).

    Returns:
        New configuration with requested automatic model fields replaced by
        concrete values.

    Raises:
        ConfigError: If automatic fields are requested for a model that has no
            resolver or if the resolver does not produce a requested field.
    """
    config_raw = config.as_dict()
    model_config = config_raw["model"]
    model_type = model_config["model_type"]
    auto_fields = _requested_auto_fields(model_config)

    if not auto_fields:
        return Config(config_raw)

    resolver = _resolver_for(model_type)

    batchprocessor = BatchProcessorRegistry.create((dataset.dataset_type, model_type), encoding=encoding)
    inputs = batchprocessor.input_preparation_single(dataset, 0)
    target = batchprocessor.target_preparation_single(dataset, 0)
    element = batchprocessor.element_preparation_single(dataset, 0)
    # Map raw inputs/target into the spaces the model actually consumes and
    # predicts. Forward processors encode the target (spectrum) and leave the
    # input untouched; inverse processors encode the input (spectrum) and
    # leave the descriptor target untouched.
    inputs = batchprocessor.encode_input(inputs, element)
    target = batchprocessor.encode_target(target, element)
    resolved_fields = resolver(inputs, target)

    for field in sorted(auto_fields):
        if field not in resolved_fields:
            raise ConfigError(f"Automatic resolver for model '{model_type}' did not produce model.{field}.")

        model_config[field] = resolved_fields[field]
        logging.info(
            "Resolved model.%s=%s from dataset '%s' with %s.",
            field,
            _format_resolved_value(resolved_fields[field]),
            dataset.dataset_type,
            type(batchprocessor).__name__,
        )

    return Config(config_raw)


###############################################################################
############################### MODEL RESOLVERS ###############################
###############################################################################


def _resolve_mlp(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve MLP input and output dimensions.

    Args:
        inputs: Prepared model input dictionary.
        target: Prepared target tensor.

    Returns:
        Mapping with MLP automatic fields.
    """
    return {
        "in_size": _last_dim(inputs["x"]),
        "out_size": _last_dim(target),
    }


def _resolve_envembed(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve EnvEmbed descriptor and spectral-basis dimensions.

    Args:
        inputs: Prepared model input dictionary.
        target: Prepared target tensor. Included for resolver interface
            consistency.

    Returns:
        Mapping with EnvEmbed automatic fields.
    """
    return {
        "in_size": _last_dim(inputs["descriptor_features"]),
        "kgroups": _kgroups_from_basis(inputs["basis"]),
    }


def _resolve_schnet(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve SchNet output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with SchNet automatic fields.
    """
    return {"reduce_channels_2": _last_dim(target)}


def _resolve_dimenet(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve DimeNet output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with DimeNet automatic fields.
    """
    return {"out_channels": _last_dim(target)}


def _resolve_dimenet_pp(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve DimeNet++ output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with DimeNet++ automatic fields.
    """
    return {"out_channels": _last_dim(target)}


def _resolve_gemnet(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve GemNet output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with GemNet automatic fields.
    """
    return {"num_targets": _last_dim(target)}


def _resolve_gemnet_oc(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve GemNet-OC output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with GemNet-OC automatic fields.
    """
    return {"num_targets": _last_dim(target)}


def _resolve_e3ee(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve E3EE output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with E3EE automatic fields.
    """
    return {"out_size": _last_dim(target)}


def _resolve_e3ee_full(inputs: dict[str, Any], target: Any) -> dict[str, Any]:
    """Resolve E3EEFull output dimension.

    Args:
        inputs: Prepared model input dictionary. Included for resolver interface
            consistency.
        target: Prepared target tensor.

    Returns:
        Mapping with E3EEFull automatic fields.
    """
    return {"out_size": _last_dim(target)}


MODEL_AUTO_RESOLVERS: dict[str, AutoResolver] = {
    "mlp": _resolve_mlp,
    "envembed": _resolve_envembed,
    "schnet": _resolve_schnet,
    "dimenet": _resolve_dimenet,
    "dimenet++": _resolve_dimenet_pp,
    "gemnet": _resolve_gemnet,
    "gemnet_oc": _resolve_gemnet_oc,
    "e3ee": _resolve_e3ee,
    "e3ee_full": _resolve_e3ee_full,
}


###############################################################################
############################# ENCODING RESOLVERS ##############################
###############################################################################


def _resolve_gaussian_encoding(item: dict[str, Any], statistics: "_TargetStatisticsCollector") -> dict[str, Any]:
    """Resolve the Gaussian-encoding spectral grid size.

    Args:
        item: Raw Gaussian encoding configuration dictionary. Included for
            resolver interface consistency.
        statistics: Streaming statistics of the training targets.

    Returns:
        Mapping with Gaussian-encoding automatic fields.
    """
    return {"num_points": statistics.overall.num_points}


def _resolve_zscore_encoding(item: dict[str, Any], statistics: "_TargetStatisticsCollector") -> dict[str, Any]:
    """Resolve z-score statistics from the training targets.

    Args:
        item: Raw z-score encoding configuration dictionary. The ``per_point``
            flag selects per-point (default) or global scalar statistics, and
            ``per_element`` selects per-element statistics keyed by absorbing
            element; when ``per_element`` is true ``elements`` may be an
            explicit list or ``"auto"``.
        statistics: Streaming statistics of the training targets.

    Returns:
        Mapping with ``mean`` and population ``std``. When ``per_element`` is
        false these are per-point lists (or single-element lists when
        ``per_point`` is false); when true they are one row per element together
        with the resolved ``elements``.
    """
    per_point = item.get("per_point", True)
    if item.get("per_element", False):
        elements = _target_elements(item, statistics)
        mean_rows: list[list[float]] = []
        std_rows: list[list[float]] = []
        for z in elements:
            stats = _require_element_stats(statistics, z, "z_score")
            if per_point:
                mean_rows.append(stats.mean.tolist())
                std_rows.append(stats.std.tolist())
            else:
                mean_rows.append([stats.global_mean])
                std_rows.append([stats.global_std])
        return {"elements": list(elements), "mean": mean_rows, "std": std_rows}

    stats = statistics.overall
    if per_point:
        return {"mean": stats.mean.tolist(), "std": stats.std.tolist()}
    return {"mean": [stats.global_mean], "std": [stats.global_std]}


def _resolve_minmax_encoding(item: dict[str, Any], statistics: "_TargetStatisticsCollector") -> dict[str, Any]:
    """Resolve minima and maxima from the training targets.

    Args:
        item: Raw min-max encoding configuration dictionary. The ``per_point``
            flag selects per-point (default) or global scalar bounds, and
            ``per_element`` selects per-element bounds keyed by absorbing
            element; when ``per_element`` is true ``elements`` may be an
            explicit list or ``"auto"``.
        statistics: Streaming statistics of the training targets.

    Returns:
        Mapping with ``minimum`` and ``maximum``. When ``per_element`` is false
        these are per-point lists (or single-element lists when ``per_point`` is
        false); when true they are one row per element together with the
        resolved ``elements``.
    """
    per_point = item.get("per_point", True)
    if item.get("per_element", False):
        elements = _target_elements(item, statistics)
        minimum_rows: list[list[float]] = []
        maximum_rows: list[list[float]] = []
        for z in elements:
            stats = _require_element_stats(statistics, z, "min_max")
            if per_point:
                minimum_rows.append(stats.minimum.tolist())
                maximum_rows.append(stats.maximum.tolist())
            else:
                minimum_rows.append([stats.global_minimum])
                maximum_rows.append([stats.global_maximum])
        return {"elements": list(elements), "minimum": minimum_rows, "maximum": maximum_rows}

    stats = statistics.overall
    if per_point:
        return {"minimum": stats.minimum.tolist(), "maximum": stats.maximum.tolist()}
    return {"minimum": [stats.global_minimum], "maximum": [stats.global_maximum]}


def _resolve_scale_encoding(item: dict[str, Any], statistics: "_TargetStatisticsCollector") -> dict[str, Any]:
    """Resolve the scaling factor from the training targets.

    The factor is the population standard deviation, so encoding divides the
    spectra to unit variance.

    Args:
        item: Raw scale encoding configuration dictionary. The ``per_point``
            flag selects per-point (default) or global scalar factors, and
            ``per_element`` selects per-element factors keyed by absorbing
            element; when ``per_element`` is true ``elements`` may be an
            explicit list or ``"auto"``.
        statistics: Streaming statistics of the training targets.

    Returns:
        Mapping with ``factor``. When ``per_element`` is false this is a
        per-point list (or single-element list when ``per_point`` is false);
        when true it is one row per element together with the resolved
        ``elements``.
    """
    per_point = item.get("per_point", True)
    if item.get("per_element", False):
        elements = _target_elements(item, statistics)
        factor_rows: list[list[float]] = []
        for z in elements:
            stats = _require_element_stats(statistics, z, "scale")
            factor_rows.append(stats.std.tolist() if per_point else [stats.global_std])
        return {"elements": list(elements), "factor": factor_rows}

    stats = statistics.overall
    if per_point:
        return {"factor": stats.std.tolist()}
    return {"factor": [stats.global_std]}


def _resolve_subtract_average_encoding(
    item: dict[str, Any], statistics: "_TargetStatisticsCollector"
) -> dict[str, Any]:
    """Resolve the per-point average spectrum from the training targets.

    Args:
        item: Raw subtract-average encoding configuration dictionary. The
            ``per_element`` flag selects per-element averages keyed by absorbing
            element; when true ``elements`` may be an explicit list or
            ``"auto"``.
        statistics: Streaming statistics of the training targets.

    Returns:
        Mapping with the per-point ``average``. When ``per_element`` is false
        this is a single per-point list; when true it is one row per element
        together with the resolved ``elements``.
    """
    if item.get("per_element", False):
        elements = _target_elements(item, statistics)
        average_rows: list[list[float]] = []
        for z in elements:
            stats = _require_element_stats(statistics, z, "subtract_average")
            average_rows.append(stats.mean.tolist())
        return {"elements": list(elements), "average": average_rows}

    return {"average": statistics.overall.mean.tolist()}


ENCODING_AUTO_RESOLVERS: dict[str, AutoResolver] = {
    "gaussian": _resolve_gaussian_encoding,
    "z_score": _resolve_zscore_encoding,
    "min_max": _resolve_minmax_encoding,
    "scale": _resolve_scale_encoding,
    "subtract_average": _resolve_subtract_average_encoding,
}

###############################################################################
################################### HELPERS ###################################
###############################################################################


def _format_resolved_value(value: Any) -> str:
    """Format an auto-resolved value for concise log output.

    Recursively truncates lists longer than 8 elements at every nesting
    level, keeping the first 3 entries and appending a length summary.
    Scalars are rendered as-is.

    Args:
        value: Resolved field value (scalar, list, or nested list).

    Returns:
        A log-friendly string representation.
    """
    return _format_value(value)


def _format_value(value: Any) -> str:
    """Recursive formatter for a single value or nested structure."""
    if isinstance(value, list):
        return _format_list(value)
    if isinstance(value, dict):
        return _format_dict(value)
    return str(value)


def _format_list(lst: list[Any]) -> str:
    """Format a list, truncating when longer than 8 elements."""
    if len(lst) <= 8:
        return "[" + ", ".join(_format_value(v) for v in lst) + "]"

    head = ", ".join(_format_value(v) for v in lst[:3])
    if all(isinstance(v, list) for v in lst):
        # Nested list of rows -- add row count and inner length summary.
        inner_lens = {len(v) for v in lst}
        lens_str = f"inner_len={inner_lens.pop()}" if len(inner_lens) == 1 else f"inner_lens={sorted(inner_lens)}"
        return f"[{head}, ...]  (rows={len(lst)}, {lens_str})"
    return f"[{head}, ...]  (len={len(lst)})"


def _format_dict(dct: dict[str, Any]) -> str:
    """Format a dict, truncating long values."""
    if len(dct) <= 4:
        items = ", ".join(f"{k}={_format_value(v)}" for k, v in dct.items())
        return "{" + items + "}"
    keys = list(dct.keys())
    items = ", ".join(f"{k}={_format_value(dct[k])}" for k in keys[:3])
    return "{" + f"{items}, ...  (keys={len(dct)})" + "}"


def _requested_auto_fields(model_config: dict[str, Any]) -> set[str]:
    """Return top-level model fields whose value is ``"auto"``.

    Args:
        model_config: Raw model configuration dictionary.

    Returns:
        Set of top-level model field names requesting automatic resolution.
    """
    return {key for key, value in model_config.items() if _is_auto(value)}


def _resolver_for(model_type: str) -> AutoResolver:
    """Return the automatic-field resolver for ``model_type``.

    Args:
        model_type: Model registry key.

    Returns:
        Resolver function for the model type.

    Raises:
        ConfigError: If no automatic-field resolver is registered for the
            model type.
    """
    try:
        return MODEL_AUTO_RESOLVERS[model_type]
    except KeyError as exc:
        raise ConfigError(f"No automatic model config resolver registered for model '{model_type}'.") from exc


def _encoding_resolver_for(encoding_type: str) -> AutoResolver:
    """Return the automatic-field resolver for ``encoding_type``.

    Args:
        encoding_type: Encoding registry key.

    Returns:
        Resolver function for the encoding type.

    Raises:
        ConfigError: If no automatic-field resolver is registered for the
            encoding type.
    """
    try:
        return ENCODING_AUTO_RESOLVERS[encoding_type]
    except KeyError as exc:
        raise ConfigError(f"No automatic encoding config resolver registered for encoding '{encoding_type}'.") from exc


def _is_auto(value: Any) -> bool:
    """Return whether ``value`` requests automatic resolution.

    Args:
        value: Raw model config value to inspect.

    Returns:
        ``True`` when ``value`` is the case-insensitive string ``"auto"``.
    """
    return isinstance(value, str) and value.lower() == AUTO_VALUE


def _last_dim(value: Any) -> int:
    """Return the final dimension of a prepared tensor-like value.

    Args:
        value: Prepared value whose final dimension should be used.

    Returns:
        Size of the final dimension.
    """
    return int(value.shape[-1])


def _collect_target_statistics(dataset: Dataset, batchprocessor: BatchProcessor) -> "_TargetStatisticsCollector":
    """Accumulate training-target statistics in a single pass.

    Iterates the training subset (or the whole dataset when no split is
    configured) and folds each prepared target into a running
    :class:`_TargetStatisticsCollector`, which maintains both overall
    statistics and per-absorbing-element statistics. Only fixed-size
    accumulators are held in memory, so the full target matrix is never
    materialized.

    Args:
        dataset: Prepared training dataset.
        batchprocessor: Batch processor for the dataset/model pair.

    Returns:
        Streaming statistics over the training targets.
    """
    subset = dataset.train_subset
    indices = list(subset.indices) if subset is not None else range(len(dataset))

    collector = _TargetStatisticsCollector()
    for index in indices:
        targets = batchprocessor.target_preparation_single(dataset, index)
        elements = batchprocessor.element_preparation_single(dataset, index)
        collector.update(targets, elements)
    return collector


def _target_elements(item: dict[str, Any], statistics: "_TargetStatisticsCollector") -> list[int]:
    """Return the atomic numbers an element-aware encoding should cover.

    Args:
        item: Raw element-aware encoding configuration dictionary. When
            ``elements`` is an explicit list it is used verbatim; otherwise all
            elements observed in the training targets are used.
        statistics: Streaming statistics of the training targets.

    Returns:
        Atomic numbers, sorted ascending when derived automatically.
    """
    requested = item.get("elements")
    if isinstance(requested, list):
        return [int(z) for z in requested]
    return statistics.sorted_elements()


def _require_element_stats(
    statistics: "_TargetStatisticsCollector", element: int, encoding_type: str
) -> "_TargetStatistics":
    """Return per-element statistics or raise when the element is absent.

    Args:
        statistics: Streaming statistics of the training targets.
        element: Atomic number to look up.
        encoding_type: Encoding identifier used in the error message.

    Returns:
        Statistics accumulated for the requested element.

    Raises:
        ConfigError: If no training target carries the requested element.
    """
    stats = statistics.per_element.get(int(element))
    if stats is None:
        raise ConfigError(f"{encoding_type} requested element {element} but no training targets carry it.")
    return stats


class _TargetStatisticsCollector:
    """Overall and per-element streaming statistics over training targets.

    Maintains one :class:`_TargetStatistics` accumulator across all targets and
    one accumulator per absorbing element. Target rows are routed to their
    element bucket using the atomic numbers supplied alongside each batch; when
    no element information is available only the overall accumulator is updated.
    """

    def __init__(self) -> None:
        """Initialize empty overall and per-element accumulators."""
        self.overall = _TargetStatistics()
        self.per_element: dict[int, _TargetStatistics] = {}

    def update(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> None:
        """Fold one batch of target spectra into the accumulators.

        Args:
            targets: Prepared target spectra ``(B, N)``.
            elements: Per-row absorber atomic numbers ``(B,)``, or ``None`` when
                the batch carries no element information.
        """
        self.overall.update(targets)
        if elements is None:
            return

        atomic_numbers = elements.to(dtype=torch.int64).reshape(-1)
        for z in torch.unique(atomic_numbers).tolist():
            mask = atomic_numbers == z
            self.per_element.setdefault(int(z), _TargetStatistics()).update(targets[mask])

    def sorted_elements(self) -> list[int]:
        """Return the observed atomic numbers in ascending order."""
        return sorted(self.per_element)


class _TargetStatistics:
    """Streaming per-point statistics over training target spectra.

    Folds batches of target spectra ``(B, N)`` into fixed-size per-point
    accumulators (count, sum, sum of squares, running minimum, and running
    maximum), so dataset-dependent encoding parameters can be derived in a
    single pass without materializing the full target matrix. Sums are
    accumulated in double precision for numerical stability.
    """

    def __init__(self) -> None:
        """Initialize an empty accumulator."""
        self._count = 0
        self._sum = torch.empty(0, dtype=torch.float64)
        self._sum_sq = torch.empty(0, dtype=torch.float64)
        self._minimum = torch.empty(0, dtype=torch.float64)
        self._maximum = torch.empty(0, dtype=torch.float64)

    def update(self, targets: torch.Tensor) -> None:
        """Fold one batch of target spectra into the accumulators.

        Args:
            targets: Prepared target spectra ``(B, N)``.
        """
        values = targets.to(dtype=torch.float64)
        batch_sum = values.sum(dim=0)
        batch_sum_sq = values.square().sum(dim=0)
        batch_min = values.amin(dim=0)
        batch_max = values.amax(dim=0)

        if self._count == 0:
            self._sum = batch_sum
            self._sum_sq = batch_sum_sq
            self._minimum = batch_min
            self._maximum = batch_max
        else:
            self._sum += batch_sum
            self._sum_sq += batch_sum_sq
            self._minimum = torch.minimum(self._minimum, batch_min)
            self._maximum = torch.maximum(self._maximum, batch_max)

        self._count += values.shape[0]

    @property
    def num_points(self) -> int:
        """Number of points ``N`` in the accumulated spectra."""
        return int(self._sum.shape[-1])

    @property
    def mean(self) -> torch.Tensor:
        """Per-point mean of the accumulated spectra."""
        return self._sum / self._count

    @property
    def std(self) -> torch.Tensor:
        """Per-point population standard deviation of the accumulated spectra."""
        variance = self._sum_sq / self._count - self.mean.square()
        return variance.clamp_min(0.0).sqrt()

    @property
    def minimum(self) -> torch.Tensor:
        """Per-point minimum of the accumulated spectra."""
        return self._minimum

    @property
    def maximum(self) -> torch.Tensor:
        """Per-point maximum of the accumulated spectra."""
        return self._maximum

    @property
    def global_mean(self) -> float:
        """Scalar mean over every point of the accumulated spectra."""
        return float(self._sum.sum() / (self._count * self.num_points))

    @property
    def global_std(self) -> float:
        """Scalar population standard deviation over every point."""
        total = self._count * self.num_points
        mean = self._sum.sum() / total
        variance = self._sum_sq.sum() / total - mean.square()
        return float(variance.clamp_min(0.0).sqrt())

    @property
    def global_minimum(self) -> float:
        """Scalar minimum over every point of the accumulated spectra."""
        return float(self._minimum.min())

    @property
    def global_maximum(self) -> float:
        """Scalar maximum over every point of the accumulated spectra."""
        return float(self._maximum.max())


def _kgroups_from_basis(basis: Any) -> list[int]:
    """Derive EnvEmbed coefficient group sizes from a spectral basis.

    Args:
        basis: Spectral basis object returned by the EnvEmbed batch processor.

    Returns:
        List of coefficient counts, one per spectral-basis width group.
    """
    num_groups = len(basis.widths_eV)
    num_coefficients = int(basis.Phi.shape[1])
    return [num_coefficients // num_groups] * num_groups
