# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Automatic model-configuration resolution from prepared datasets."""

import logging
from collections.abc import Callable
from typing import Any

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.datasets import Dataset
from xanesnet.utils.exceptions import ConfigError

from .config import Config

AUTO_VALUE = "auto"
AutoResolver = Callable[[dict[str, Any], Any], dict[str, Any]]


def resolve_auto_model_config(config: Config, dataset: Dataset) -> Config:
    """Resolve model fields set to ``"auto"`` from a prepared dataset.

    The input config is expected to have passed schema-backed training
    validation, including validation that ``"auto"`` is used only for supported
    top-level model fields. This function only performs the dataset-dependent
    finalization: it uses the registered batch processor for the dataset/model
    pair to prepare one sample, asks the model-specific resolver for concrete
    dimensions, and returns a new ``Config`` without mutating the input config.

    Args:
        config: Validated training configuration.
        dataset: Prepared training dataset used to derive automatic model
            values.

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

    batchprocessor = BatchProcessorRegistry.create((dataset.dataset_type, model_type))
    inputs = batchprocessor.input_preparation_single(dataset, 0)
    target = batchprocessor.target_preparation_single(dataset, 0)
    resolved_fields = resolver(inputs, target)

    for field in sorted(auto_fields):
        if field not in resolved_fields:
            raise ConfigError(f"Automatic resolver for model '{model_type}' did not produce model.{field}.")

        model_config[field] = resolved_fields[field]
        logging.info(
            "Resolved model.%s=%s from dataset '%s' with %s.",
            field,
            resolved_fields[field],
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
################################### HELPERS ###################################
###############################################################################


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
