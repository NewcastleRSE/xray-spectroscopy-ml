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

import torch
import random


def train_data(
        mode: str,
        model_mode: str,
        x_path: str,
        y_path: str,
        descriptor_type: str,
        descriptor_params: dict = {},
        data_params: dict = {},
        kfold_params: dict = {},
        hyperparams: dict = {},
        max_samples: int = None,
        variance_threshold: float = 0.0,
        epochs: int = 100,
        callbacks: dict = {},
        seed: int = None,
        save: bool = True,
        bootstrap: dict = {},
        ensemble: dict = {},
        fourier_transform: bool = False,
):
    rng = RandomState(seed=seed)

    xyz_path = [Path(p) for p in glob(x_path)]
    xanes_path = [Path(p) for p in glob(y_path)]

    xyz_list = []
    xanes_list = []
    e_list = []
    element_label = []

    for n_element in range(0, len(xyz_path)):
        element_name = str(xyz_path[n_element]).split("/")[-3]

        for path in (xyz_path[n_element], xanes_path[n_element]):
            if not path.exists():
                err_str = f"path to X/Y data ({path}) doesn't exist"
                raise FileNotFoundError(err_str)

        if xyz_path[n_element].is_dir() and xanes_path[n_element].is_dir():
            print(">> loading data from directories...\n")

            ids = list(
                set(list_filestems(xyz_path[n_element]))
                & set(list_filestems(xanes_path[n_element]))
            )

            ids.sort()

            descriptors = {"rdc": RDC, "wacsf": WACSF}

            descriptor = descriptors.get(descriptor_type)(**descriptor_params)

            n_samples = len(ids)
            n_x_features = descriptor.get_len()
            n_y_features = linecount(
                xanes_path[n_element] / f"{ids[0]}.txt") - 2

            xyz_data = np.full((n_samples, n_x_features), np.nan)
            print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
            xanes_data = np.full((n_samples, n_y_features), np.nan)
            print(">> preallocated {}x{} array for Y data...".format(
                *xanes_data.shape))
            print(">> ...everything preallocated!\n")

            print(">> loading data into array(s)...")
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                element_label.append(element_name)
                with open(xyz_path[n_element] / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                xyz_data[i, :] = descriptor.transform(atoms)
                with open(xanes_path[n_element] / f"{id_}.txt", "r") as f:
                    xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
            print(">> ...loaded into array(s)!\n")

            xyz_list.append(xyz_data)
            xanes_list.append(xanes_data)
            e_list.append(e)

        elif x_path[n_element].is_file() and y_path[n_element].is_file():
            print(">> loading data from .npz archive(s)...\n")

            with open(x_path[n_element], "rb") as f:
                xyz_data = np.load(f)["x"]
            print(">> ...loaded {}x{} array of X data".format(*xyz_data.shape))
            with open(y_path[n_element], "rb") as f:
                xanes_data = np.load(f)["y"]
                e = np.load(f)["e"]
            print(">> ...loaded {}x{} array of Y data".format(*xanes_data.shape))
            print(">> ...everything loaded!\n")

            xyz_list.append(xyz_data)
            xanes_list.append(xanes_data)
            e_list.append(e)

            if save:
                print(">> overriding save flag (running in `--no-save` mode)\n")
                save = False

        else:
            err_str = (
                "paths to X/Y data are expected to be either a) both "
                "files (.npz archives), or b) both directories"
            )
            raise TypeError(err_str)

    xyz_data = np.vstack(xyz_list)
    xanes_data = np.vstack(xanes_list)
    e = np.vstack(e_list)
    element_label = np.asarray(element_label)

    # Transform data
    if fourier_transform:
        from data_transform import fourier_transform_data
        print('>> Transforming training data using Fourier transform...')
        xanes_data = fourier_transform_data(xanes_data)

    # DATA AUGMENTATION
    if data_params:
        from data_augmentation import data_augment

        xyz_data, xanes_data = data_augment(
            data_params, n_samples, n_x_features, n_y_features
        )

    print(">> shuffling and selecting data...")
    xyz, xanes, element = shuffle(
        xyz_data, xanes_data, element_label, random_state=rng, n_samples=max_samples
    )
    print(">> ...shuffled and selected!\n")

    if bootstrap["fn"] == "True":
        from bootstrap_fn import bootstrap_train

        data_compress = {"ids": ids, "x": xyz_data, "y": xanes_data, "e": e}

        bootstrap_train(
            bootstrap,
            xyz,
            xanes,
            mode,
            model_mode,
            hyperparams,
            epochs,
            save,
            kfold_params,
            rng,
            descriptor,
            data_compress,
        )

    elif ensemble["fn"] == "True":

        from ensemble_fn import ensemble_train

        data_compress = {"ids": ids, "x": xyz_data, "y": xanes_data, "e": e}

        ensemble_train(
            ensemble,
            xyz,
            xanes,
            mode,
            model_mode,
            hyperparams,
            epochs,
            save,
            kfold_params,
            rng,
            descriptor,
            data_compress,
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
                hyperparams,
                epochs,
                kfold_params,
                rng,
                hyperparams["weight_init_seed"],
            )

        elif mode == "train_xanes":
            from core_learn import train_xanes

            model = train_xanes(
                xyz,
                xanes,
                exp_name,
                model_mode,
                hyperparams,
                epochs,
                kfold_params,
                rng,
                hyperparams["weight_init_seed"],
            )

        elif mode == "train_aegan":
            from core_learn import train_aegan

            model = train_aegan(
                xyz, xanes, exp_name, model_mode, hyperparams, epochs, kfold_params, rng
            )

        if save:
            parent_model_dir = "model/"
            Path(parent_model_dir).mkdir(parents=True, exist_ok=True)

            model_dir = unique_path(Path(parent_model_dir), "model")
            model_dir.mkdir()

            with open(model_dir / "descriptor.pickle", "wb") as f:
                pickle.dump(descriptor, f)
            with open(model_dir / "dataset.npz", "wb") as f:
                np.savez_compressed(f, ids=ids, x=xyz_data, y=xanes_data, e=e)

            torch.save(model, model_dir / f"model.pt")
            print("Saved model to disk")
            json.dump(
                descriptor_params, open(
                    f"{model_dir}/{descriptor_type}.txt", "w")
            )

        else:
            print("none")

    return
