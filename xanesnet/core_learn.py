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

import logging
import random
import sys
import torch
import time

from datetime import timedelta
from enum import Enum
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from torch import nn
from torchinfo import summary

from xanesnet.models.pre_trained import PretrainedModels
from xanesnet.utils.switch import KernelInitSwitch, BiasInitSwitch
from xanesnet.utils.io import (
    save_models,
    load_pretrained_descriptors,
    load_pretrained_model,
)
from xanesnet.creator import (
    create_learn_scheme,
    create_descriptors,
    create_model,
    create_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        # logging.FileHandler("train.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)


class TrainingMode(Enum):
    XYZ_TO_XANES = "train_xyz"
    XANES_TO_XYZ = "train_xanes"
    TRAIN_ALL = "train_all"


def train(config, args):
    """
    Train ML model based on the provided configuration and arguments.
    """
    try:
        mode = TrainingMode(args.mode)
        logging.info(f">> Training mode: {mode.value}")
    except ValueError:
        raise ValueError(f"'{args.mode}' is not a valid training mode.")

    # Setup descriptors from inputscript or pretrained model
    descriptor_list = _setup_descriptors(config)

    # Load, encode, and preprocess data
    dataset = _setup_datasets(config, descriptor_list)

    # Assign dataset items to training features X and labels y
    X, y = _setup_X_y(config, mode, dataset)

    # Print dataset summary
    _dataset_summary(config, X, y)

    # Setup model from inputscript or pretrained model
    model = _setup_model(config, X, y, mode)

    # Setup training scheme
    scheme = _setup_scheme(config, args, model, X, y)

    # Run model training
    model_list, scheme_type, train_time = _train_models(config, scheme)

    # Print trained model summary
    _model_summary(config, model_list[0], X, y)

    # Print training time
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

    # Save model, encoded data and config to disk
    if args.save:
        metadata = {
            "mode": args.mode,
            "dataset": dataset.config,
            "model": model.config,
            "descriptors": [desc.config for desc in descriptor_list],
            "scheme": scheme_type,
            "standardscaler": config["standardscaler"],
        }

        save_models(Path("models"), model_list, metadata)


def _setup_descriptors(config):
    """Initialises or loads descriptors."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading descriptors from pretrained model: {model_type}")
        descriptor_list = load_pretrained_descriptors(model_type)
    else:
        descriptor_config = config["descriptors"]
        descriptor_types = ", ".join(d["type"] for d in descriptor_config)
        logging.info(f">> Initialising descriptors: {descriptor_types}")
        descriptor_list = create_descriptors(config=descriptor_config)

    return descriptor_list


def _setup_datasets(config, descriptor_list):
    dataset_type = config["dataset"]["type"]

    logging.info(f">> Initialising training datasets: {dataset_type}")
    # Pack kwargs
    kwargs = {
        "root": config["dataset"]["root_path"],
        "xyz_path": config["dataset"]["xyz_path"],
        "xanes_path": config["dataset"]["xanes_path"],
        "descriptors": descriptor_list,
        "shuffle": True,
        **config["dataset"].get("params", {}),
    }

    dataset = create_dataset(dataset_type, **kwargs)
    return dataset


def _setup_X_y(config, mode: TrainingMode, dataset):
    logging.info(f">> Setting X and y datasets ...")

    X = y = None
    if mode in [TrainingMode.XYZ_TO_XANES, TrainingMode.TRAIN_ALL]:
        logging.info(f">> X = XYZ dataset, Y = Xanes dataset")
        X, y = dataset.xyz_data, dataset.xanes_data

    elif mode == TrainingMode.XANES_TO_XYZ:
        logging.info(f">> X = Xanes dataset, Y = XYZ dataset")
        X, y = dataset.xanes_data, dataset.xyz_data

    if config.get("standardscaler"):
        logging.info(">> Applying standard scaler to X...")
        X = StandardScaler().fit_transform(X)

    return X, y


def _setup_model(config, X, y, mode: TrainingMode):
    """Initialises or loads the model and its descriptors."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading pretrained model: {model_type}")
        model_params = config["model"].get("params", {})
        model = load_pretrained_model(model_type, **model_params)
    else:
        logging.info(f">> Initialising model: {model_type}")
        model_params = config["model"].get("params", {})

        # Add additional model parameters
        if model_type.lower() == "gnn":
            model_params["in_size"] = X[0].x.shape[1]
            model_params["out_size"] = X[0].y.shape[0]
            model_params["mlp_feat_size"] = X[0].graph_attr.shape[0]
        else:
            model_params["in_size"] = X.shape[1]
            model_params["out_size"] = y.shape[1]

        model = create_model(model_type, **model_params)

        weights_config = config["model"].get("weights", {})
        weight_kernel = weights_config.get("kernel", "xavier_uniform")
        logging.info(f">> Initialising model weights: {weight_kernel}")
        model = _init_model_weights(model, **weights_config)

    return model


def _init_model_weights(model, **kwargs):
    """
    Initialise model weights and biases
    """
    kernel = kwargs.get("kernel", "xavier_uniform")
    bias = kwargs.get("bias", "zeros")
    seed = kwargs.get("seed", random.randrange(1000))

    # set seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    kernel_init_fn = KernelInitSwitch().get(kernel)
    bias_init_fn = BiasInitSwitch().get(bias)

    # nested function to apply to each module
    def _init_fn(m):
        # Initialise Conv and Linear layers
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            kernel_init_fn(m.weight)
            if m.bias is not None:
                bias_init_fn(m.bias)

    model.apply(_init_fn)

    return model


def _setup_scheme(config, args, model, X, y):
    model_type = config["model"].get("type")

    logging.info(">> Initialising training scheme")
    # Pack kwargs
    kwargs = {
        "model_config": config.get("model"),
        "hyper_params": config.get("hyperparams", {}),
        "kfold_params": config.get("kfold_params", {}),
        "bootstrap_params": config.get("bootstrap_params", {}),
        "ensemble_params": config.get("ensemble_params", {}),
        "lr_scheduler": config.get("lr_scheduler", False),
        "scheduler_params": config.get("scheduler_params", {}),
        "mlflow": args.mlflow,
        "tensorboard": args.tensorboard,
    }

    scheme = create_learn_scheme(model_type, model, X=X, y=y, **kwargs)

    return scheme


def _train_models(config, scheme):
    model_list = []
    start_time = time.time()
    if config["bootstrap"]:
        logging.info(">> Training model using bootstrap resampling...\n")
        scheme_type = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        logging.info(">> Training model using ensemble learning...\n")
        scheme_type = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        logging.info(">> Training model using kfold cross-validation...\n")
        scheme_type = "kfold"
        model_list.append(scheme.train_kfold())
    else:
        logging.info(">> Training model using standard training procedure...\n")
        scheme_type = "std"
        model_list.append(scheme.train_std())

    train_time = time.time() - start_time

    return model_list, scheme_type, train_time


def _dataset_summary(config, X, y):
    # Print dataset summary
    model_type = config["model"]["type"]
    if model_type.lower() == "gnn":
        logging.info(
            f">> Graph dataset (samples: {len(X)}, "
            f"node features: {X[0].x.shape[1]}, "
            f"edge features: {X[0].edge_attr.shape[1]}, "
            f"graph features: {X[0].graph_attr.shape[0]})"
        )
    else:
        logging.info(f">> XYZ dataset: samples = {X.shape[0]}, features = {X.shape[1]}")
        logging.info(
            f">> XANES dataset: samples = {y.shape[0]}, features = {y.shape[1]}"
        )


def _model_summary(config, model, X, y):
    logging.info("\n--- Model Summary ---")
    model_type = config["model"].get("type")

    if model_type.lower() == "aegan_mlp":
        X_dim = X.shape[1]
        y_dim = y.shape[1]
        dummy_x = torch.randn(1, X_dim)
        dummy_y = torch.randn(1, y_dim)
        input_data = (dummy_x, dummy_y)
    elif model_type.lower() == "gnn":
        input_data = None
    else:
        X_dim = X.shape[1]
        dummy_x = torch.randn(1, X_dim)
        input_data = dummy_x

    summary(model, input_data=input_data)
