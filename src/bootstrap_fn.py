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
from model_utils import make_dir, model_mode_error
from predict import predict_xanes, predict_xyz, y_predict_dim
from spectrum.xanes import XANES
from utils import unique_path
from inout import save_xanes_mean


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
    kfold,
    kfold_params,
    rng,
    descriptor,
    data_compress,
    lr_scheduler,
    model_eval,
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
                hyperparams["weight_init_seed"],
                lr_scheduler,
                model_eval,
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
                hyperparams["weight_init_seed"],
                lr_scheduler,
                model_eval,
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
                hyperparams["weight_init_seed"],
                lr_scheduler,
                model_eval,
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

    for i in range(1, n_boot + 1):
        n_dir = f"{model_dir}/model_{i:03d}/model.pt"

        model = torch.load(n_dir, map_location=torch.device("cpu"))
        model.eval()
        print("Loaded model from disk")

        if model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
            parent_model_dir, predict_dir = make_dir()
        else:
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

                for id_, y_predict_ in tqdm.tqdm(zip(ids, y_predict)):
                    with open(predict_dir / f"{id_}.txt", "w") as f:
                        f.write(
                            "\n".join(
                                map(str, y_predict_.detach().numpy()))
                        )

            elif mode == "predict_xanes":
                xanes_predict = predict_xanes(xyz_data, model)

                x = xyz_data
                y = xanes_data
                y_predict = xanes_predict

                if fourier_transform:
                    y_predict = data_transform.inverse_fourier_transform_data(
                        y_predict)

                # print(y_predict.detach().numpy().shape)
                # print(np.asarray(e).shape)

                # for id_, y_predict_ in tqdm.tqdm(zip(ids, y_predict)):
                #     with open(predict_dir / f"{id_}.txt", "w") as f:
                #         save_xanes(
                #             f, XANES(np.asarray(e), y_predict_.detach().numpy()))

            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )
            y_predict, e = y_predict_dim(y_predict, ids, model_dir)
            print(e.shape)
            print(y_predict.shape)

            if plot_save:
                plot.plot_predict(ids, y, y_predict, e, predict_dir, mode)

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

            y_predict_score.append(mean_squared_error(
                y, y_predict.detach().numpy()))
            x_recon_score.append(mean_squared_error(
                x, x_recon.detach().numpy()))

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

        elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
            # Convert to float
            if config["x_path"] is not None and config["y_path"] is not None:
                x = torch.tensor(xyz_data).float()
                y = torch.tensor(xanes_data).float()
            elif config["x_path"] is not None and config["y_path"] is None:
                x = torch.tensor(xyz_data).float()
                y = None
            elif config["y_path"] is not None and config["x_path"] is None:
                y = torch.tensor(xanes_data).float()
                x = None

            import aegan_predict

            x_recon, y_predict, y_recon, x_predict = aegan_predict.main(
                config,
                x,
                y,
                model,
                fourier_transform,
                model_dir,
                predict_dir,
                ids,
                parent_model_dir,
            )

            if config["x_path"] is not None:
                x_recon_score.append(mean_squared_error(x, x_recon))

            if config["y_path"] is not None:
                y_recon_score.append(mean_squared_error(y, y_recon))

            if config["x_path"] is not None and config["y_path"] is not None:
                y_predict_score.append(mean_squared_error(y, y_predict))
                x_predict_score.append(mean_squared_error(x, x_predict))

    if model_mode == "mlp" or model_mode == "cnn":
        mean_score = torch.mean(torch.tensor(y_predict_score))
        std_score = torch.std(torch.tensor(y_predict_score))
        print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}")

        y_predict_all = np.asarray(y_predict_all)
        mean_y_predict = np.mean(y_predict_all, axis=0)
        std_y_predict = np.std(y_predict_all, axis=0)

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
