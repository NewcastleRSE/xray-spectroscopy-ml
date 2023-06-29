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
import os
import pickle as pickle
import random
from pathlib import Path

import numpy as np
import torch
import tqdm as tqdm
from sklearn.metrics import mean_squared_error

import data_transform
import plot
from model_utils import make_dir
from predict import predict_xanes, predict_xyz, y_predict_dim
from spectrum.xanes import XANES
from utils import unique_path
from inout import save_xanes_mean, save_xyz_mean


def bootstrap_data(xyz, xanes, n_size, seed):
    random.seed(seed)

    new_xyz = []
    new_xanes = []

    for i in range(int(xyz.shape[0] * n_size)):
        idx = random.randint(0, xyz.shape[0]-1)
        new_xyz.append(xyz[idx])
        new_xanes.append(xanes[idx])

    return np.asarray(new_xyz), np.asarray(new_xanes)


def bootstrap_train(
    bootstrap,
    xyz,
    xanes,
    mode,
    model_mode,
    hyperparams,
    epochs,
    save,
    kfold,
    kfold_params,
    rng,
    descriptor,
    data_compress,
    lr_scheduler,
    model_eval,
    load_guess,
    loadguess_params,
    optuna_params,
):
    parent_bootstrap_dir = "bootstrap/"
    Path(parent_bootstrap_dir).mkdir(parents=True, exist_ok=True)

    bootstrap_dir = unique_path(Path(parent_bootstrap_dir), "bootstrap")
    bootstrap_dir.mkdir()

    # getting exp name for mlflow
    exp_name = f"{mode}_{model_mode}"

    for i in range(bootstrap["n_boot"]):
        n_xyz, n_xanes = bootstrap_data(
            xyz, xanes, bootstrap["n_size"], bootstrap["seed_boot"][i]
        )
        print(n_xyz.shape)
        if mode == "train_xyz":
            from core_learn import train_xyz

            model = train_xyz(
                n_xyz,
                n_xanes,
                exp_name,
                model_mode,
                hyperparams,
                epochs,
                kfold,
                kfold_params,
                rng,
                bootstrap["seed_boot"][i],
                lr_scheduler,
                model_eval,
                load_guess,
                loadguess_params,
                optuna_params
            )
        elif mode == "train_xanes":
            from core_learn import train_xanes

            model = train_xanes(
                n_xyz,
                n_xanes,
                exp_name,
                model_mode,
                hyperparams,
                epochs,
                kfold,
                kfold_params,
                rng,
                bootstrap["seed_boot"][i],
                lr_scheduler,
                model_eval,
                load_guess,
                loadguess_params,
                optuna_params
            )

        elif mode == "train_aegan":
            from core_learn import train_aegan

            model = train_aegan(
                n_xyz,
                n_xanes,
                exp_name,
                model_mode,
                hyperparams,
                epochs,
                kfold,
                kfold_params,
                rng,
                bootstrap["seed_boot"][i],
                lr_scheduler,
                model_eval,
                load_guess,
                loadguess_params,
                optuna_params
            )
        if save:
            with open(bootstrap_dir / "descriptor.pickle", "wb") as f:
                pickle.dump(descriptor, f)
            with open(bootstrap_dir / "dataset.npz", "wb") as f:
                np.savez_compressed(
                    f,
                    ids=data_compress["ids"],
                    x=data_compress["x"],
                    y=data_compress["y"],
                )

            model_dir = unique_path(Path(bootstrap_dir), "model")
            model_dir.mkdir()
            torch.save(model, model_dir / f"model.pt")


def bootstrap_predict(
    model_dir,
    mode,
    model_mode,
    xyz_data,
    xanes_data,
    e,
    ids,
    plot_save,
    fourier_transform,
    config,
):
    n_boot = len(next(os.walk(model_dir))[1])

    x_recon_score = []
    y_predict_score = []
    y_recon_score = []
    x_predict_score = []
    y_predict_all = []

    parent_model_dir, predict_dir = make_dir()

    for i in range(1, n_boot + 1):
        n_dir = f"{model_dir}/model_{i:03d}/model.pt"

        model = torch.load(n_dir, map_location=torch.device("cpu"))
        model.eval()
        print("Loaded model from disk")

        if model_mode == "mlp" or model_mode == "cnn":
            if mode == "predict_xyz":
                if fourier_transform:
                    xanes_data = data_transform.fourier_transform_data(
                        xanes_data)

                xyz_predict = predict_xyz(xanes_data, model)

                x = xanes_data
                y = xyz_data
                y_predict = xyz_predict

            elif mode == "predict_xanes":
                xanes_predict = predict_xanes(xyz_data, model)

                x = xyz_data
                y = xanes_data
                y_predict = xanes_predict

                if fourier_transform:
                    y_predict = data_transform.inverse_fourier_transform_data(
                        y_predict)
                    
            if y is not None:
                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_predict.detach().numpy()),
                )
            y_predict = y_predict_dim(y_predict, ids)

            if plot_save:
                plot.plot_predict(ids, y, y_predict, predict_dir, mode)

            if y is not None:
                y_predict_score.append(mean_squared_error(
                    y, y_predict.detach().numpy()))

            y_predict_all.append(y_predict.detach().numpy())

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            if mode == "predict_xyz":
                x = xanes_data

                if fourier_transform:
                    xanes_data = data_transform.fourier_transform_data(
                        xanes_data)

                recon_xanes, pred_xyz = predict_xyz(xanes_data, model)

                x_recon = recon_xanes
                y = xyz_data
                y_predict = pred_xyz

                if fourier_transform:
                    x_recon = data_transform.inverse_fourier_transform_data(
                        x_recon)

            elif mode == "predict_xanes":
                recon_xyz, pred_xanes = predict_xanes(xyz_data, model)

                x = xyz_data
                x_recon = recon_xyz
                y = xanes_data
                y_predict = pred_xanes

                if fourier_transform:
                    y_predict = data_transform.inverse_fourier_transform_data(
                        y_predict)

            if y is not None:
                y_predict_score.append(mean_squared_error(
                    y, y_predict.detach().numpy()))
                
            x_recon_score.append(mean_squared_error(
                x, x_recon.detach().numpy()))
            
            y_predict_all.append(y_predict.detach().numpy())

            print(
                "MSE x to x recon : ",
                mean_squared_error(x, x_recon.detach().numpy()),
            )
            if y is not None:
                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_predict.detach().numpy()),
                )
            y_predict = y_predict_dim(y_predict, ids)
            if plot_save:
                plot.plot_ae_predict(
                    ids, y, y_predict, x, x_recon, e, predict_dir, mode
                )

        elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":

            x = xyz_data
            y = xanes_data

            import aegan_predict

            x_recon, y_pred, y_recon, x_pred = aegan_predict.predict_aegan(x, y, model, mode, fourier_transform)

            if x is not None and x_recon is not None:
                x_recon_score.append(mean_squared_error(x, x_recon.detach().numpy()))

            if x is not None and x_pred is not None:
                x_predict_score.append(mean_squared_error(x, x_pred.detach().numpy()))

            if y is not None and y_recon is not None:
                y_recon_score.append(mean_squared_error(y, y_recon.detach().numpy()))

            if y is not None and y_pred is not None:
                y_predict_score.append(mean_squared_error(y, y_pred.detach().numpy()))

    if model_mode == "mlp" or model_mode == "cnn":
        mean_score = torch.mean(torch.tensor(y_predict_score))
        std_score = torch.std(torch.tensor(y_predict_score))
        print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}")

        y_predict_all = np.asarray(y_predict_all)
        mean_y_predict = np.mean(y_predict_all, axis=0)
        std_y_predict = np.std(y_predict_all, axis=0)

        if y is None:
            # Dummy array for e
            e = np.arange(y_predict.shape[1])

        if mode == "predict_xyz":
            for id_, mean_y_predict_, std_y_predict_ in tqdm.tqdm(zip(ids, mean_y_predict, std_y_predict)):
                with open(predict_dir / f"{id_}.txt", "w") as f:
                    save_xyz_mean(f, mean_y_predict_, std_y_predict_)

        elif mode == "predict_xanes":
            for id_, mean_y_predict_, std_y_predict_ in tqdm.tqdm(zip(ids, mean_y_predict, std_y_predict)):
                with open(predict_dir / f"{id_}.txt", "w") as f:
                    save_xanes_mean(
                        f, XANES(e, mean_y_predict_), std_y_predict_)

    elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
        mean_score = torch.mean(torch.tensor(y_predict_score))
        std_score = torch.std(torch.tensor(y_predict_score))
        print(
            f"Mean score predict: {mean_score:.4f}, Std score: {std_score:.4f}")
        mean_score = torch.mean(torch.tensor(x_recon_score))
        std_score = torch.std(torch.tensor(x_recon_score))
        print(
            f"Mean score reconstruction: {mean_score:.4f}, Std score: {std_score:.4f}"
        )

        y_predict_all = np.asarray(y_predict_all)
        mean_y_predict = np.mean(y_predict_all, axis=0)
        std_y_predict = np.std(y_predict_all, axis=0)

        if y is None:
            # Dummy array for e
            e = np.arange(y_predict.shape[1])

        if mode == "predict_xyz":
            for id_, mean_y_predict_, std_y_predict_ in tqdm.tqdm(zip(ids, mean_y_predict, std_y_predict)):
                with open(predict_dir / f"{id_}.txt", "w") as f:
                    save_xyz_mean(f, mean_y_predict_, std_y_predict_)

        elif mode == "predict_xanes":
            for id_, mean_y_predict_, std_y_predict_ in tqdm.tqdm(zip(ids, mean_y_predict, std_y_predict)):
                with open(predict_dir / f"{id_}.txt", "w") as f:
                    save_xanes_mean(
                        f, XANES(e, mean_y_predict_), std_y_predict_)

    elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
        if config["x_path"] is not None:
            mean_score = torch.mean(torch.tensor(x_recon_score))
            std_score = torch.std(torch.tensor(x_recon_score))
            print(
                f"Mean score x reconstruction: {mean_score:.4f}, Std score: {std_score:.4f}"
            )
        if config["y_path"] is not None:
            mean_score = torch.mean(torch.tensor(y_recon_score))
            std_score = torch.std(torch.tensor(y_recon_score))
            print(
                f"Mean score y reconstruction: {mean_score:.4f}, Std score: {std_score:.4f}"
            )
        if config["x_path"] is not None and config["y_path"] is not None:
            mean_score = torch.mean(torch.tensor(y_predict_score))
            std_score = torch.std(torch.tensor(y_predict_score))
            print(
                f"Mean y prediction score: {mean_score:.4f}, Std score: {std_score:.4f}"
            )
            mean_score = torch.mean(torch.tensor(x_predict_score))
            std_score = torch.std(torch.tensor(x_predict_score))
            print(
                f"Mean x prediction score: {mean_score:.4f}, Std score: {std_score:.4f}"
            )
