# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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

"""Lightning training through ``train_initial`` and ``train_worker`` entry points."""

import importlib
import json
import logging
import os
from argparse import Namespace
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from xanesnet.core_train import (
    _run_training,
    _save_results,
    _setup_dataset,
    _setup_datasource,
    _setup_encoding,
    _setup_strategy,
)
from xanesnet.serialization.auto_config import (
    resolve_auto_encoding_config,
    resolve_auto_model_config,
)
from xanesnet.serialization.config import Config
from xanesnet.serialization.model_profile import (
    reset_peak_memory_stats,
)
from xanesnet.serialization.splits import save_split_indices
from xanesnet.utils.distributed import (
    LIGHTNING_CONTEXT_ENV,
    LIGHTNING_LAUNCH_ENVIRONMENT,
    get_lightning_context_path,
)
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.logger import setup_file_logging
from xanesnet.utils.random import set_global_seed

_CONTEXT_NAME = "lightning_context.json"


def _register_lightning_trainer() -> None:
    """Load the optional trainer module so it registers in this process."""
    importlib.import_module("xanesnet.runners.trainers.lightning")


###############################################################################
################################## LIGHTNING ##################################
###############################################################################


def train_initial(config: Config, args_namespace: Namespace, save_dir: Path) -> None:
    """Prepare one Lightning run, execute training, and export its models.

    Builds the data source and dataset from ``config``, resolves any automatic
    fields, prepares the encoding, builds the strategy from the resolved config,
    and publishes the run context before ``fit()`` starts workers.
    Only this initial process exports shared models and reports.

    Args:
        config: Validated training configuration with the effective seed set
            by the CLI, possibly containing supported ``"auto"`` fields.
        args_namespace: Parsed CLI arguments (must contain ``tensorboard``;
            may contain ``dry_run`` to enable profile output).
        save_dir: Root directory for all training outputs.
    """
    logging.info("Lighting training.")

    _register_lightning_trainer()

    save_dir = save_dir.resolve()
    config_raw = config.as_dict()
    config_raw["dataset"]["root"] = str(Path(config_raw["dataset"]["root"]).resolve())
    config = Config(config_raw)

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)

    logging.info("Resolving 'auto' encoding fields...")
    config = resolve_auto_encoding_config(config, dataset)

    encoding = _setup_encoding(config, dataset)

    logging.info("Resolving 'auto' model fields...")
    config = resolve_auto_model_config(config, dataset, encoding)
    resolved_config_save_path = config.save(save_dir / "resolved_train_config.yaml")
    logging.info(f"Resolved training config saved to: {resolved_config_save_path}.")

    strategy = _setup_strategy(config, dataset, encoding, save_dir, args_namespace.tensorboard)
    strategy.setup_models()
    strategy.setup_checkpointer()
    strategy.init_model_weights()
    strategy.setup_trainers(config.get_str("device"))

    # Save signature
    signature = Config(
        {
            "dataset": dataset.signature,
            "model": strategy.model_signature,
            "strategy": strategy.signature,
            "encodings": encoding.signature,
        }
    )
    signature_save_path = signature.save(save_dir / "models" / "signature.yaml")
    logging.info(f"Signature saved to: {signature_save_path}")

    if strategy.checkpointer is not None:
        strategy.checkpointer.set_signature(signature)

    # Save split indices if they were generated
    split_path = save_dir / "split_indices.json"
    save_split_indices(split_path, dataset.get_all_subset_indices())
    logging.info(f"Split indices saved to: {split_path}")

    # Build lighting context
    processes = config.section("trainer").get_int("num_gpus") if config.get_str("device").startswith("cuda") else 1
    context = Config(
        {
            "version": 1,
            "run_id": uuid4().hex,
            "owner_pid": os.getpid(),
            "processes": processes,
            "save_dir": str(save_dir),
            "config": config.as_dict(),
            "dataset_length": len(dataset),
            "dataset_signature": dataset.signature.as_dict(),
            "split_digest": sha256(split_path.read_bytes()).hexdigest(),
            "tensorboard": args_namespace.tensorboard,
            "dry_run": args_namespace.dry_run,
            "active": True,
        }
    )

    # Main training
    with _published_context(context):
        if args_namespace.dry_run:
            reset_peak_memory_stats(config.get_str("device"))

        model_list, train_time = _run_training(strategy)

        _save_results(model_list, train_time, config, dataset, encoding, signature, save_dir, args_namespace.dry_run)


def train_worker() -> None:
    """Reconstruct a prepared Lightning run and train without shared writes.

    The original user YAML is not reloaded. Configuration, seed, and split
    identity come from the initial process's active context.

    Raises:
        ConfigError: If bootstrap context, process role, or prepared data is
            missing or inconsistent.
    """
    context = _load_worker_context()
    run_id = context.get_str("run_id")
    _register_lightning_trainer()
    config = context.section("config")
    save_dir = Path(context.get_str("save_dir"))
    rank = _environment_int("LOCAL_RANK")
    worker_log_dir = save_dir / "workers" / f"rank_{rank}"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(worker_log_dir)
    logging.info(f"Lightning worker {rank} loading prepared run {run_id}.")
    set_global_seed(config.get_int("seed"))

    datasource = _setup_datasource(config)
    worker_config_raw = config.as_dict()
    worker_dataset_config = worker_config_raw["dataset"]
    worker_dataset_config["skip_prepare"] = True
    worker_dataset_config["split_indexfile"] = str(save_dir / "split_indices.json")
    worker_config = Config(worker_config_raw)
    dataset = _setup_dataset(worker_config, datasource)
    if (
        len(dataset) != context.get_int("dataset_length")
        or dataset.signature.as_dict() != context.section("dataset_signature").as_dict()
    ):
        raise ConfigError("Prepared dataset does not match the Lightning run context.")
    encoding = _setup_encoding(config, dataset)
    strategy = _setup_strategy(config, dataset, encoding, save_dir, False)
    strategy.setup_models()
    strategy.setup_checkpointer()
    strategy.init_model_weights()
    strategy.setup_trainers(config.get_str("device"))
    if context.get_bool("dry_run"):
        reset_peak_memory_stats(config.get_str("device"))
    _run_training(strategy)


###############################################################################
############################### WORKER CONTEXT ################################
###############################################################################


def _environment_int(name: str) -> int:
    """Read an integer required by the supported distributed launcher.

    Args:
        name: Required environment variable name.

    Returns:
        Parsed integer value.

    Raises:
        ConfigError: If the variable is missing or is not an integer.
    """
    try:
        return int(os.environ[name])
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"Lightning worker requires an integer {name}.") from exc


def _load_worker_context() -> Config:
    """Validate worker identity and load its immutable prepared configuration.

    Returns:
        Active context belonging to the current self-launched worker.

    Raises:
        ConfigError: If context files, owner identity, rank, or prepared split
            contents are missing or inconsistent.
    """
    path = get_lightning_context_path()
    if path is None:
        raise ConfigError("Lightning worker has no prepared-run context.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot load Lightning context: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Lightning context must contain an object.")
    context = Config(payload)
    if context.get_int("version") != 1 or not context.get_bool("active"):
        raise ConfigError("Lightning context is unsupported or no longer active.")

    owner_pid = context.get_int("owner_pid")
    rank = _environment_int("LOCAL_RANK")
    processes = context.get_int("processes")
    if owner_pid <= 0 or owner_pid == os.getpid() or not 0 < rank < processes:
        raise ConfigError("Lightning context does not identify this process as a worker.")
    if _environment_int("WORLD_SIZE") != processes or _environment_int("NODE_RANK") != 0:
        raise ConfigError("Lightning context requires the expected single-node worker group.")
    if "RANK" in os.environ and _environment_int("RANK") != rank:
        raise ConfigError("Lightning worker rank does not match its single-node launch.")
    try:
        os.kill(owner_pid, 0)
    except OSError as exc:
        raise ConfigError("The initial Lightning process is no longer available.") from exc

    save_dir = Path(context.get_str("save_dir"))
    if path != save_dir / _CONTEXT_NAME:
        raise ConfigError("Lightning context has an inconsistent run directory.")
    split_path = save_dir / "split_indices.json"
    try:
        digest = sha256(split_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigError("Prepared Lightning split indices are missing.") from exc
    if digest != context.get_str("split_digest"):
        raise ConfigError("Prepared Lightning split indices have changed.")
    return context


@contextmanager
def _published_context(context: Config) -> Generator[None, None, None]:
    """Publish active worker context and restore launcher state on every exit.

    Args:
        context: Complete initial-process context to publish before ``fit()``.

    Yields:
        Control while workers may inherit the context marker.
    """
    path = Path(context.get_str("save_dir")) / _CONTEXT_NAME
    names = (*LIGHTNING_LAUNCH_ENVIRONMENT, LIGHTNING_CONTEXT_ENV)
    previous = {name: os.environ.get(name) for name in names}
    _write_context(path, context)
    try:
        os.environ[LIGHTNING_CONTEXT_ENV] = str(path)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        payload = context.as_dict()
        payload["active"] = False
        _write_context(path, Config(payload))


def _write_context(path: Path, context: Config) -> None:
    """Atomically replace a context file after writing complete JSON data.

    Args:
        path: Final context path inside the run directory.
        context: JSON-compatible context values.
    """
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(context.as_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
