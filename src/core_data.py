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
import numpy as np
import pickle as pickle
import tqdm as tqdm
import json

from pathlib import Path
from glob import glob
from numpy.random import RandomState
from sklearn.utils import shuffle

from inout import load_xyz
from inout import load_xanes

from utils import unique_path
from utils import linecount
from utils import list_filestems
from structure.rdc import RDC
from structure.wacsf import WACSF
from structure.soap import SOAP
from structure.mbtr import MBTR
from structure.lmbtr import LMBTR

import torch
import random


def train_data(mode: str, model_mode: str, config, save: bool = True, fourier_transform: bool = False, max_samples: int = None):
    rng = RandomState(seed=config["seed"])

    xyz_path = Path(config["x_path"])
    xanes_path = Path(config["y_path"])

    for path in (xyz_path, xanes_path):

        for path in (xyz_path, xanes_path):
            if not path.exists():
                err_str = f"path to X/Y data ({path}) doesn't exist"
                raise FileNotFoundError(err_str)

    if xyz_path.is_dir() and xanes_path.is_dir():

        ids = list(
            set(list_filestems(xyz_path))
            & set(list_filestems(xanes_path))
        )

        ids.sort()

        descriptors = {"rdc": RDC, "wacsf": WACSF,
                       "soap": SOAP, "mbtr": MBTR, "lmbtr": LMBTR}

        descriptor = descriptors.get(config["descriptor"]["type"])(
            **config["descriptor"]["params"]
        )

        n_samples = len(ids)
        if config["descriptor"]["type"] == 'wacsf' or config["descriptor"]["type"] == 'rdc':
            n_x_features = descriptor.get_len()
        else:
            n_x_features = descriptor.get_number_of_features()
        n_y_features = linecount(
            xanes_path / f"{ids[0]}.txt") - 2

        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}x{} array for Y data...".format(
            *xanes_data.shape))
        print(">> ...everything preallocated!\n")

        if config["descriptor"]["type"] == 'wacsf' or config["descriptor"]["type"] == 'rdc':
            print(">> loading data into array(s)...")
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                xyz_data[i, :] = descriptor.transform(atoms)
                with open(xanes_path / f"{id_}.txt", "r") as f:
                    xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
            print(">> ...loaded into array(s)!\n")
        elif config["descriptor"]["type"] == 'mbtr':
            print(">> loading data into array(s)...")
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                    tmp = descriptor.create(atoms)
                xyz_data[i, :] = tmp
                with open(xanes_path / f"{id_}.txt", "r") as f:
                    xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
            print(">> ...loaded into array(s)!\n")
        elif config["descriptor"]["type"] == 'lmbtr':
            print(">> loading data into array(s)...")
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                    tmp = descriptor.create(atoms, positions=[0])
                xyz_data[i, :] = tmp
                with open(xanes_path / f"{id_}.txt", "r") as f:
                    xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
            print(">> ...loaded into array(s)!\n")
        elif config["descriptor"]["type"] == 'soap':
            print(">> loading data into array(s)...")
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                    tmp = descriptor.create_single(atoms, positions=[0])
                xyz_data[i, :] = tmp
                with open(xanes_path / f"{id_}.txt", "r") as f:
                    xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
            print(">> ...loaded into array(s)!\n")
        else:
            print(">> ...This descriptor doesn't exist, try again!!\n")

    elif xyz_path.is_file() and xanes_path.is_file():
        print(">> loading data from .npz archive(s)...\n")

        with open(xyz_path, "rb") as f:
            xyz_data = np.load(f)["x"]
        print(">> ...loaded {}x{} array of X data".format(*xyz_data.shape))
        with open(xanes_path, "rb") as f:
            xanes_data = np.load(f)["y"]
            e = np.load(f)["e"]
        print(">> ...loaded {}x{} array of Y data".format(*xanes_data.shape))
        print(">> ...everything loaded!\n")

        if save:
            print(">> overriding save flag (running in `--no-save` mode)\n")
            save = False

    else:
        err_str = (
            "paths to X/Y data are expected to be either a) both "
            "files (.npz archives), or b) both directories"
        )
        raise TypeError(err_str)

    xyz, xanes = shuffle(xyz_data, xanes_data,
                         random_state=rng, n_samples=max_samples)

    # Transform data
    if fourier_transform:
        from data_transform import fourier_transform_data

        print(">> Transforming training data using Fourier transform...")
        xanes = fourier_transform_data(xanes)

    # DATA AUGMENTATION
    if config["data_params"]:
        from data_augmentation import data_augment

        xyz, xanes = data_augment(
            config["augment"], xyz, xanes, n_samples, n_x_features, n_y_features
        )

    if config["bootstrap"]:
        from bootstrap_fn import bootstrap_train

        data_compress = {"ids": ids, "x": xyz_data, "y": xanes_data}

        bootstrap_train(
            config["bootstrap_params"],
            xyz,
            xanes,
            mode,
            model_mode,
            config["hyperparams"],
            config["epochs"],
            save,
            config["kfold"],
            config["kfold_params"],
            rng,
            descriptor,
            data_compress,
            config["lr_scheduler"],
            config["model_eval"],
            config["load_guess"],
            config["loadguess_params"],
            config["optuna_params"]
        )

    elif config["ensemble"]:
        from ensemble_fn import ensemble_train

        data_compress = {"ids": ids, "x": xyz_data, "y": xanes_data}

        ensemble_train(
            config["ensemble_params"],
            xyz,
            xanes,
            mode,
            model_mode,
            config["hyperparams"],
            config["epochs"],
            save,
            config["kfold"],
            config["kfold_params"],
            rng,
            descriptor,
            data_compress,
            config["lr_scheduler"],
            config["model_eval"],
            config["load_guess"],
            config["loadguess_params"],
            config["optuna_params"]
        )

    else:
        # getting exp name for mlflow
        exp_name = f"{mode}_{model_mode}"
        if mode == "train_xyz":
            from core_learn import train_xyz

            model = train_xyz(
                xyz,
                xanes,
                exp_name,
                model_mode,
                config["hyperparams"],
                config["epochs"],
                config["kfold"],
                config["kfold_params"],
                rng,
                config["hyperparams"]["weight_init_seed"],
                config["lr_scheduler"],
                config["model_eval"],
                config["load_guess"],
                config["loadguess_params"],
                config["optuna_params"],
            )

        elif mode == "train_xanes":
            from core_learn import train_xanes

            model = train_xanes(
                xyz,
                xanes,
                exp_name,
                model_mode,
                config["hyperparams"],
                config["epochs"],
                config["kfold"],
                config["kfold_params"],
                rng,
                config["hyperparams"]["weight_init_seed"],
                config["lr_scheduler"],
                config["model_eval"],
                config["load_guess"],
                config["loadguess_params"],
                config["optuna_params"],
            )

        elif mode == "train_aegan":
            from core_learn import train_aegan

            model = train_aegan(
                xyz,
                xanes,
                exp_name,
                model_mode,
                config["hyperparams"],
                config["epochs"],
                config["kfold"],
                config["kfold_params"],
                rng,
                config["hyperparams"]["weight_init_seed"],
                config["lr_scheduler"],
                config["model_eval"],
                config["load_guess"],
                config["loadguess_params"],
                config["optuna_params"],
            )

        if save:
            parent_model_dir = "model/"
            Path(parent_model_dir).mkdir(parents=True, exist_ok=True)

            model_dir = unique_path(Path(parent_model_dir), "model")
            model_dir.mkdir()

            with open(model_dir / "descriptor.pickle", "wb") as f:
                pickle.dump(descriptor, f)
            with open(model_dir / "dataset.npz", "wb") as f:
                np.savez_compressed(f, ids=ids, x=xyz_data, y=xanes_data)

            torch.save(model, model_dir / f"model.pt")
            print("Saved model to disk")
            descriptor_type = config["descriptor"]["type"]
            json.dump(
                config["descriptor"]["params"],
                open(f"{model_dir}/{descriptor_type}.txt", "w"),
            )
        else:
            print("none")

    return
