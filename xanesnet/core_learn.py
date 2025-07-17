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
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch import nn
from torchinfo import summary

from xanesnet.encoder import data_learn, data_gnn_learn
from xanesnet.utils.fourier import fourier_transform
from xanesnet.models import AEGAN_MLP, GNN
from xanesnet.models.pre_trained import PretrainedModels
from xanesnet.utils.switch import DataAugmentSwitch, KernelInitSwitch, BiasInitSwitch
from xanesnet.utils.io import save_models
from xanesnet.creator import (
    create_learn_scheme,
    create_descriptors,
    create_model,
    create_pretrained_model,
    create_pretrained_descriptors,
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
    AEGAN = "train_aegan"


def train(config, args):
    """
    Train ML model based on the provided configuration and arguments.
    """
    try:
        mode = TrainingMode(args.mode)
    except ValueError:
        raise ValueError(f"'{args.mode}' is not a valid training mode.")

    # Setup descriptors from inputscript or pretrained model
    descriptor_list = _setup_descriptors(config)

    # Load, encode, and preprocess data
    X, y, index = _setup_datasets(config, descriptor_list, mode)

    # Setup model from inputscript or pretrained model
    model = _setup_model(config, X, y)

    # Setup training scheme
    scheme = _setup_scheme(config, args, model, X, y)

    # Run model training
    model_list, train_scheme, train_time = _train_models(config, scheme)

    # Print model summary
    _model_summary(model_list[0], X, y)

    # Print training time
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")

    # Save model, encoded data and config to disk
    if args.save:
        metadata = {
            "mode": args.mode,
            "model": model.config,
            "descriptors": [desc.config for desc in descriptor_list],
            "scheme": train_scheme,
            "standardscaler": config["standardscaler"],
            "fourier_transform": config["fourier_transform"],
            "fourier_param": config["fourier_params"],
            "node_features": config.get("node_features", {}),
            "edge_features": config.get("edge_features", {}),
        }

        dataset = {"index": index, "X": X, "y": y}
        save_models(Path("models"), model_list, metadata, dataset=dataset)


def _setup_descriptors(config):
    """Initialises or loads descriptors."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading descriptors from pretrained model: {model_type}")
        descriptor_list = create_pretrained_descriptors(model_type)
    else:
        descriptor_config = config["descriptors"]
        descriptor_types = ", ".join(d["type"] for d in descriptor_config)
        logging.info(f">> Initialising descriptors: {descriptor_types}")
        descriptor_list = create_descriptors(config=descriptor_config)

    return descriptor_list


def _setup_datasets(config, descriptor_list, mode: TrainingMode):
    model_type = config["model"]["type"]

    logging.info(">> Encoding training datasets...")
    if model_type.lower() == "gnn":
        # Preprocessing datasets into graph representation for GNN input
        X, y, index = data_gnn_learn(
            config["xyz_path"],
            config["xanes_path"],
            config["node_features"],
            config["edge_features"],
            descriptor_list,
            config["fourier_transform"],
            config["fourier_params"],
        )
    else:
        # Preprocessing datasets for non-GNN models
        xyz, xanes, index = data_learn(
            config["xyz_path"], config["xanes_path"], descriptor_list
        )

        n_samples = config.get("hyperparams", {}).get("n_samples")
        seed = config.get("hyperparams", {}).get("seed", random.sample(range(1000), 1))
        logging.info(
            f">> Shuffling training datasets: n_samples = {n_samples or 'all'}"
        )
        xyz, xanes = shuffle(
            xyz, xanes, random_state=RandomState(seed=seed), n_samples=n_samples
        )

        if config.get("fourier_transform"):
            logging.info("Applying Fourier transform to spectra data...")
            params = config.get("fourier_params", {})
            xanes = fourier_transform(xanes, params.get("concat", False))

        if config.get("data_augment"):
            logging.info("Applying data augmentation...")
            params = config.get("augment_params", {})
            xyz, xanes = DataAugmentSwitch().augment(xyz, xanes, **params)

        # Assigns the final X and Y datasets based on the training mode.
        logging.info(f">> Setting X and Y datasets for mode: {mode.value}")

        if mode in [TrainingMode.XYZ_TO_XANES, TrainingMode.AEGAN]:
            X, y = xyz, xanes
            logging.info(f">> X = xyz (samples: {X.shape[0]}, features: {X.shape[1]})")
            logging.info(f">> y = xanes(samples: {y.shape[0]}, features: {y.shape[1]})")
        else:
            X, y = xanes, xyz
            logging.info(
                f">> X = xanes (samples: {X.shape[0]}, features: {X.shape[1]})"
            )
            logging.info(f">> y = xyz(samples: {y.shape[0]}, features: {y.shape[1]})")

        if config.get("standardscaler"):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    return X, y, index


def _setup_model(config, X, y):
    """Initialises or loads the model and its descriptors."""
    model_type = config["model"]["type"]

    if hasattr(PretrainedModels, model_type):
        logging.info(f">> Loading pretrained model: {model_type}")
        model_params = config["model"].get("params", {})
        model = create_pretrained_model(model_type, **model_params)
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
        train_scheme = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        logging.info(">> Training model using ensemble learning...\n")
        train_scheme = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        logging.info(">> Training model using kfold cross-validation...\n")
        train_scheme = "kfold"
        model_list.append(scheme.train_kfold())
    else:
        logging.info(">> Training model using standard training procedure...\n")
        train_scheme = "std"
        model_list.append(scheme.train_std())

    train_time = time.time() - start_time

    return model_list, train_scheme, train_time


def _model_summary(model, X, y):
    logging.info("\n--- Model Summary ---")

    if isinstance(model, AEGAN_MLP):
        X_dim = X.shape[1]
        y_dim = y.shape[1]
        dummy_x = torch.randn(1, X_dim)
        dummy_y = torch.randn(1, y_dim)
        input_data = (dummy_x, dummy_y)
    elif isinstance(model, GNN):
        input_data = None
    else:
        X_dim = X.shape[1]
        dummy_x = torch.randn(1, X_dim)
        input_data = dummy_x

    summary(model, input_data=input_data)
