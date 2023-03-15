import os
import pickle as pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import plot
from model_utils import model_mode_error
from predict import predict_xanes, predict_xyz, y_predict_dim
from utils import unique_path
import data_transform


def bootstrap_data(xyz, xanes, n_size, seed):
    random.seed(seed)

    new_xyz = []
    new_xanes = []

    for i in range(int(xyz.shape[0] * n_size)):
        new_xyz.append(random.choice(xyz))
        new_xanes.append(random.choice(xanes))

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
    kfold_params,
    rng,
    descriptor,
    data_compress,
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
                kfold_params,
                rng,
                hyperparams["weight_init_seed"],
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
                kfold_params,
                rng,
                hyperparams["weight_init_seed"],
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
                kfold_params,
                rng,
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
                    e=data_compress["e"],
                )

            model_dir = unique_path(Path(bootstrap_dir), "model")
            model_dir.mkdir()
            torch.save(model, model_dir / f"model.pt")


def bootstrap_predict(
    model_dir, mode, model_mode, xyz_data, xanes_data, ids, plot_save, fourier_transform
):
    n_boot = len(next(os.walk(model_dir))[1])

    bootstrap_score = []
    for i in range(n_boot):
        n_dir = f"{model_dir}/model_00{i+1}/model.pt"

        model = torch.load(n_dir, map_location=torch.device("cpu"))
        model.eval()
        print("Loaded model from disk")

        parent_model_dir, predict_dir = model_mode_error(
            model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]
        )

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

            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )
            y_predict, e = y_predict_dim(y_predict, ids, model_dir)
            if plot_save:
                plot.plot_predict(ids, y, y_predict, e, predict_dir, mode)

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

            print(
                "MSE x to x recon : ",
                mean_squared_error(x, x_recon.detach().numpy()),
            )
            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )
            y_predict, e = y_predict_dim(y_predict, ids, model_dir)
            if plot_save:
                plot.plot_ae_predict(
                    ids, y, y_predict, x, x_recon, e, predict_dir, mode
                )

        bootstrap_score.append(mean_squared_error(
            y, y_predict.detach().numpy()))
    mean_score = torch.mean(torch.tensor(bootstrap_score))
    std_score = torch.std(torch.tensor(bootstrap_score))
    print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}")
