import os
import pickle as pickle
import numpy as np
import random
from pathlib import Path

import torch
from sklearn.metrics import mean_squared_error

import data_transform
import model_utils
from predict import predict_xanes, predict_xyz
from utils import unique_path


def ensemble_train(
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
):
    parent_ensemble_dir = "ensemble/"
    Path(parent_ensemble_dir).mkdir(parents=True, exist_ok=True)

    ensemble_dir = unique_path(Path(parent_ensemble_dir), "ensemble")
    ensemble_dir.mkdir()
    print(ensemble_dir)

    # getting exp name for mlflow
    exp_name = f"{mode}_{model_mode}"
    for i in range(ensemble["n_ens"]):

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
                ensemble["weight_init_seed"][i],
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
                ensemble["weight_init_seed"][i],
            )

        elif mode == "train_aegan":
            from core_learn import train_aegan

            model = train_aegan(
                xyz,
                xanes,
                exp_name,
                model_mode,
                hyperparams,
                epochs,
                kfold_params,
                rng,
            )
        if save:
            with open(ensemble_dir / "descriptor.pickle", "wb") as f:
                pickle.dump(descriptor, f)
            with open(ensemble_dir / "dataset.npz", "wb") as f:
                np.savez_compressed(
                    f,
                    ids=data_compress["ids"],
                    x=data_compress["x"],
                    y=data_compress["y"],
                    e=data_compress["e"],
                )

            model_dir = unique_path(Path(ensemble_dir), "model")
            model_dir.mkdir()
            torch.save(model, model_dir / f"model.pt")


def ensemble_predict(ensemble, model_dir, mode, model_mode, xyz_data, xanes_data, plot_save, fourier_transform):

    if ensemble["combine"] == "prediction":

        n_model = len(next(os.walk(model_dir))[1])

        ensemble_preds = []
        ensemble_recons = []
        for i in range(n_model):
            n_dir = f"{model_dir}/model_00{i+1}/model.pt"

            model = torch.load(n_dir, map_location=torch.device("cpu"))
            model.eval()
            print("Loaded model from disk")

            if fourier_transform:
                parent_model_dir, predict_dir = model_utils.model_mode_error(
                    model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]*2
                )
            else:
                parent_model_dir, predict_dir = model_utils.model_mode_error(
                    model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]
                )

            if model_mode == "mlp" or model_mode == "cnn":
                if mode == "predict_xyz":

                    if fourier_transform:
                        xanes_data = data_transform.fourier_transform_data(
                            xanes_data)

                    xyz_predict = predict_xyz(xanes_data, model)
                    ensemble_preds.append(xyz_predict)
                    y = xyz_data

                elif mode == "predict_xanes":
                    xanes_predict = predict_xanes(xyz_data, model)

                    if fourier_transform:
                        xanes_predict = data_transform.inverse_fourier_transform_data(
                            xanes_predict)

                    ensemble_preds.append(xanes_predict)
                    y = xanes_data

            elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
                if mode == "predict_xyz":

                    x = xanes_data
                    y = xyz_data

                    if fourier_transform:
                        xanes_data = data_transform.fourier_transform_data(
                            xanes_data)

                    xanes_recon, xyz_predict = predict_xyz(xanes_data, model)

                    if fourier_transform:
                        xanes_recon = data_transform.inverse_fourier_transform_data(
                            xanes_recon)

                    ensemble_preds.append(xyz_predict)
                    ensemble_recons.append(xanes_recon)

                elif mode == "predict_xanes":

                    x = xyz_data
                    y = xanes_data

                    xyz_recon, xanes_predict = predict_xanes(xyz_data, model)

                    if fourier_transform:
                        xanes_predict = data_transform.inverse_fourier_transform_data(
                            xanes_predict)

                    ensemble_preds.append(xanes_predict)
                    ensemble_recons.append(xyz_recon)

        ensemble_pred = sum(ensemble_preds) / len(ensemble_preds)
        print(
            "MSE y to y pred : ",
            mean_squared_error(y, ensemble_pred.detach().numpy()),
        )
        if model_mode == "ae_mlp" or model_mode == "ae_cnn":
            ensemble_recon = sum(ensemble_recons) / len(ensemble_recons)
            print(
                "MSE x to x recon : ",
                mean_squared_error(x, ensemble_recon.detach().numpy()),
            )
    elif ensemble["combine"] == "weight":
        print("ensemble by combining weight")

        n_model = len(next(os.walk(model_dir))[1])
        ensemble_model = []

        for i in range(n_model):
            n_dir = f"{model_dir}/model_00{i+1}/model.pt"

            model = torch.load(n_dir, map_location=torch.device("cpu"))
            ensemble_model.append(model)
        from model_utils import make_dir

        parent_model_dir, predict_dir = make_dir()

        if model_mode == "mlp" or model_mode == "cnn":
            from model import EnsembleModel

            model = EnsembleModel(ensemble_model)
            print("Loaded model from disk")
            if mode == "predict_xyz":

                if fourier_transform:
                    xanes_data = data_transform.fourier_transform_data(
                        xanes_data)

                y_predict = predict_xyz(xanes_data, model)
                y = xyz_data

            elif mode == "predict_xanes":

                xanes_predict = predict_xanes(xyz_data, model)

                if fourier_transform:
                    xanes_predict = data_transform.inverse_fourier_transform_data(
                        xanes_predict)

                y_predict = predict_xyz(xyz_data, model)
                y = xanes_data

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            from model import AutoencoderEnsemble

            model = AutoencoderEnsemble(ensemble_model)
            print("Loaded model from disk")
            if mode == "predict_xyz":
                x = xanes_data
                y = xyz_data

                if fourier_transform:
                    xanes_data = data_transform.fourier_transform_data(
                        xanes_data)

                x_recon, y_predict = predict_xyz(xanes_data, model)

                if fourier_transform:
                    x_recon = data_transform.inverse_fourier_transform_data(
                        x_recon)

            elif mode == "predict_xanes":
                x = xyz_data
                y = xanes_data

                x_recon, y_predict = predict_xanes(xyz_data, model)

                if fourier_transform:
                    y_predict = data_transform.inverse_fourier_transform_data(
                        y_predict)

        print(
            "MSE y to y pred : ",
            mean_squared_error(y, y_predict.detach().numpy()),
        )
        if model_mode == "ae_mlp" or model_mode == "ae_cnn":
            print(
                "MSE x to x recon : ",
                mean_squared_error(x, x_recon.detach().numpy()),
            )
