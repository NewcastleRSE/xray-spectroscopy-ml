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

from xanesnet.data_descriptor import encode_train
from xanesnet.utils import save_model_list, save_model
from xanesnet.creator import (
    create_descriptor,
    create_learn_scheme,
)


def train_model(config, args):
    """
    Train ML model based on the provided configuration and arguments
    """
    # Encode training dataset with specified descriptor type
    print(
        ">> Encoding training dataset with %s descriptor"
        % config["descriptor"]["type"],
    )

    descriptor = create_descriptor(
        config["descriptor"]["type"], **config["descriptor"]["params"]
    )

    xyz, xanes, index, n_x_features, n_y_features = encode_train(
        config["xyz_path"], config["xanes_path"], descriptor
    )

    # Shuffle the encoded data for randomness
    xyz, xanes = shuffle(
        xyz, xanes, random_state=RandomState(seed=config["hyperparams"]["seed"]), n_samples=config["hyperparams"].get("max_samples", None)
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
    scheme = create_learn_scheme(
        x_data,
        y_data,
        config["model"],
        config["hyperparams"],
        config["kfold"],
        config["kfold_params"],
        config["bootstrap_params"],
        config["ensemble_params"],
        config["lr_scheduler"],
        config["scheduler_params"],
        config["optuna"],
        config["optuna_params"],
        config["freeze"],
        config["freeze_params"],
        config["standardscaler"],
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
            "descriptor_type": config["descriptor"]["type"],
            "descriptor_param": config["descriptor"]["params"],
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
