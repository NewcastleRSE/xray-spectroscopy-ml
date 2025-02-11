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

from pathlib import Path

from numpy.random import RandomState
from sklearn.utils import shuffle

from xanesnet.data_encoding import data_learn, data_gnn_learn
from xanesnet.utils import save_models
from xanesnet.creator import (
    create_descriptor,
    create_learn_scheme,
)


def train_model(config, args):
    """
    Train ML model based on the provided configuration and arguments
    """

    # Encode training dataset with specified descriptor types
    descriptor_list = []
    descriptors = config.get("descriptors")

    if descriptors is None:
        raise ValueError("No descriptors found in the configuration file!")

    for d in descriptors:
        print(f">> Initialising {d['type']} feature descriptor...")
        if d["type"] in ("mace", "direct"):
            descriptor = create_descriptor(d["type"])
        else:
            descriptor = create_descriptor(d["type"], **d["params"])
        descriptor_list.append(descriptor)

    xyz, xanes, index = data_learn(
        config["xyz_path"], config["xanes_path"], descriptor_list
    )

    # Shuffle the encoded data for randomness
    xyz, xanes = shuffle(
        xyz,
        xanes,
        random_state=RandomState(seed=config["hyperparams"]["seed"]),
        n_samples=config["hyperparams"].get("max_samples", None),
    )
    print(
        ">> Shuffled training dataset and limited to n_samples = %s"
        % config["hyperparams"].get("max_samples", None),
    )

    # Apply FFT to spectra training dataset if specified
    if config["fourier_transform"]:
        from .data_transform import fourier_transform

        print(">> Transforming spectra data using Fourier transform...")
        xanes = fourier_transform(xanes, config["fourier_params"]["concat"])

    # Apply data augmentation if specified
    if config["data_augment"]:
        from .data_augmentation import data_augment

        print(">> Applying data augmentation...")
        xyz, xanes = data_augment(config["augment_params"], xyz, xanes)

    # assign descriptor and spectra datasets to X and Y based on train mode
    if args.mode == "train_xyz" or args.mode == "train_aegan":
        x_data = xyz
        y_data = xanes
    elif args.mode == "train_xanes":
        x_data = xanes
        y_data = xyz
    else:
        raise ValueError(f"Unsupported mode name: {args.mode}")

    # Initialise learn scheme
    print(">> Initialising learn scheme...")
    kwargs = {
        "model": config["model"],
        "hyper_params": config["hyperparams"],
        "kfold": config["kfold"],
        "kfold_params": config["kfold_params"],
        "bootstrap_params": config["bootstrap_params"],
        "ensemble_params": config["ensemble_params"],
        "scheduler": config["lr_scheduler"],
        "scheduler_params": config["scheduler_params"],
        "optuna": config["optuna"],
        "optuna_params": config["optuna_params"],
        "freeze": config["freeze"],
        "freeze_params": config["freeze_params"],
        "scaler": config["standardscaler"],
    }
    print(kwargs)
    scheme = create_learn_scheme(
        x_data,
        y_data,
        **kwargs,
    )

    models = []
    # Train the model using selected training strategy
    print(">> Training %s model..." % config["model"]["type"])
    if config["bootstrap"]:
        train_scheme = "bootstrap"
        models = scheme.train_bootstrap()
    elif config["ensemble"]:
        train_scheme = "ensemble"
        models = scheme.train_ensemble()
    elif config["kfold"]:
        train_scheme = "kfold"
        models.append(scheme.train_kfold())
    else:
        train_scheme = "std"
        models.append(scheme.train_std())

    # Save trained model, metadata, compressed dataset, and descriptors to disk
    if args.save == "yes":
        metadata = {
            "mode": args.mode,
            "model_type": config["model"]["type"],
            "descriptors": config["descriptors"],
            "hyperparams": config["hyperparams"],
            "lr_scheduler": config["scheduler_params"],
            "standardscaler": config["standardscaler"],
            "fourier_transform": config["fourier_transform"],
            "fourier_param": config["fourier_params"],
            "scheme": train_scheme,
        }

        dataset = {"ids": index, "x": xyz, "y": xanes}
        save_models(Path("models"), models, descriptor_list, metadata, dataset=dataset)


def train_model_gnn(config, args):
    if args.mode != "train_xyz":
        raise ValueError(f"Unsupported mode name for GNN: {args.mode}")

    # Encode training dataset with specified descriptor types
    descriptor_list = []
    if config["descriptors"] is not None:
        for d in config["descriptors"]:
            print(f">> Initialising {d['type']} feature descriptor...")
            descriptor = create_descriptor(d["type"], **d["params"])
            descriptor_list.append(descriptor)

    graph_dataset = data_gnn_learn(
        config["xyz_path"],
        config["xanes_path"],
        config["model"]["node_features"],
        config["model"]["edge_features"],
        descriptor_list,
    )

    # Initialise learn scheme
    print(">> Initialising learn scheme...")
    kwargs = {
        "model": config["model"],
        "hyper_params": config["hyperparams"],
        "kfold": config["kfold"],
        "kfold_params": config["kfold_params"],
        "bootstrap_params": config["bootstrap_params"],
        "ensemble_params": config["ensemble_params"],
        "scheduler": config["lr_scheduler"],
        "scheduler_params": config["scheduler_params"],
        "optuna": config["optuna"],
        "optuna_params": config["optuna_params"],
        "freeze": config["freeze"],
        "freeze_params": config["freeze_params"],
        "scaler": config["standardscaler"],
    }

    models = []
    scheme = create_learn_scheme(graph_dataset, None, **kwargs)
    # Train the model using selected training strategy
    print(">> Training %s model..." % config["model"]["type"])
    if config["bootstrap"]:
        train_scheme = "bootstrap"
        models = scheme.train_bootstrap()
    elif config["ensemble"]:
        train_scheme = "ensemble"
        models = scheme.train_ensemble()
    elif config["kfold"]:
        train_scheme = "kfold"
        models.append(scheme.train_kfold())
    else:
        train_scheme = "std"
        models.append(scheme.train_std())

    # Save trained model, metadata, and descriptors to disk
    if args.save == "yes":
        metadata = {
            "mode": args.mode,
            "model_type": config["model"]["type"],
            "node_features": config["model"]["node_features"],
            "edge_features": config["model"]["edge_features"],
            "descriptors": config["descriptors"],
            "hyperparams": config["hyperparams"],
            "lr_scheduler": config["scheduler_params"],
            "standardscaler": config["standardscaler"],
            "fourier_transform": config["fourier_transform"],
            "fourier_param": config["fourier_params"],
            "scheme": train_scheme,
        }

        save_models(Path("models"), models, descriptor_list, metadata)
