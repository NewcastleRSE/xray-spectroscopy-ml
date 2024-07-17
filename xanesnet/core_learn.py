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

from numpy.random import RandomState
from sklearn.utils import shuffle

from xanesnet.data_descriptor import encode_train, encode_train_gnn
from xanesnet.utils import save_model_list, save_model
from xanesnet.creator import (
    create_descriptor,
    create_learn_scheme,
)


def train_model(config, args):
    """
    Train ML model based on the provided configuration and arguments
    """

    if len(config["descriptors"]) > 1:
        raise ValueError("Non-GNN model can only have one descriptor.")

    # Encode training dataset with specified descriptor types
    print(
        ">> Encoding training dataset with %s descriptor"
        % config["descriptors"][0]["type"],
    )
    descriptor = create_descriptor(
        config["descriptors"][0]["type"], **config["descriptors"][0]["params"]
    )
    xyz, xanes, index, n_x_features, n_y_features = encode_train(
        config["xyz_path"], config["xanes_path"], descriptor
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
        xyz, xanes = data_augment(
            config["augment_params"],
            xyz,
            xanes,
            index,
            n_x_features,
            n_y_features,
        )

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
        "scaler": config["standardscaler"]
    }

    scheme = create_learn_scheme(
        x_data,
        y_data,
        **kwargs,
    )

    # Train the model using selected training strategy
    print(">> Training %s model..." % config["model"]["type"])
    if config["bootstrap"]:
        train_scheme = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        train_scheme = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        train_scheme = "std"
        model = scheme.train_kfold()
    else:
        train_scheme = "std"
        model = scheme.train_std()

    # Save model to file if specified
    save_path = "models/"
    if args.save == "yes":
        metadata = {
            "mode": args.mode,
            "model_type": config["model"]["type"],
            "descriptor_type": config["descriptors"][0]["type"],
            "descriptor_param": config["descriptors"][0]["params"],
            "hyperparams": config["hyperparams"],
            "lr_scheduler": config["scheduler_params"],
            "standardscaler": config["standardscaler"],
            "fourier_transform": config["fourier_transform"],
            "fourier_param": config["fourier_params"],
            "scheme": train_scheme,
        }

        data_compress = {"ids": index, "x": xyz, "y": xanes}
        if config["bootstrap"] or config["ensemble"]:
            save_model_list(
                save_path, model_list, descriptor, data_compress, metadata, config
            )
        else:
            save_model(save_path, model, descriptor, data_compress, metadata)


def train_model_gnn(config, args):
    node_descriptors = []
    edge_descriptors = []

    node_dtypes = config["model"]["node_descriptors"]
    edge_dtypes = config["model"]["edge_descriptors"]

    print(
        ">> Initialising GNN node and edge feature descriptors (node feat:",
        node_dtypes,
        "edge feat:",
        edge_dtypes,
        ")...",
    )
    # Assign descriptors to the corresponding list
    for desc in config["descriptors"]:
        descriptor = create_descriptor(desc["type"], **desc["params"])
        if desc["type"] in node_dtypes:
            node_descriptors.append(descriptor)
        if desc["type"] in edge_dtypes:
            edge_descriptors.append(descriptor)

    graph_dataset = encode_train_gnn(
        config["xyz_path"],
        config["xanes_path"],
        node_descriptors,
        edge_descriptors
    )

    if args.mode != "train_xyz":
        raise ValueError(f"Unsupported mode name for GNN: {args.mode}")

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
        "scaler": config["standardscaler"]
    }

    scheme = create_learn_scheme(
        graph_dataset,
        None,
        **kwargs
    )

    # Train the model using selected training strategy
    print(">> Training %s model..." % config["model"]["type"])
    if config["bootstrap"]:
        train_scheme = "bootstrap"
        model_list = scheme.train_bootstrap()
    elif config["ensemble"]:
        train_scheme = "ensemble"
        model_list = scheme.train_ensemble()
    elif config["kfold"]:
        train_scheme = "std"
        model = scheme.train_kfold()
    else:
        train_scheme = "std"
        model = scheme.train_std()
