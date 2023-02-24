import os
import numpy as np
import torch
import random
from pathlib import Path
import pickle as pickle
from sklearn.metrics import mean_squared_error

from utils import unique_path
from model_utils import bootstrap_fn
from model_utils import model_mode_error
import plot


# ***************** might move to new python file -- predict.py**********
def predict_xyz(xanes_data, model):
    print("predict xyz structure")
    xanes = torch.from_numpy(xanes_data)
    xanes = xanes.float()

    pred_xyz = model(xanes)

    return pred_xyz


def predict_xanes(xyz_data, model):
    print("predict xanes spectrum")
    xyz = torch.from_numpy(xyz_data)
    xyz = xyz.float()

    pred_xanes = model(xyz)

    return pred_xanes


def y_predict_dim(y_predict, ids, model_dir):
    if y_predict.ndim == 1:
        if len(ids) == 1:
            y_predict = y_predict.reshape(-1, y_predict.size)
        else:
            y_predict = y_predict.reshape(y_predict.size, -1)
    print(">> ...predicted Y data!\n")

    with open(model_dir / "dataset.npz", "rb") as f:
        e = np.load(f)["e"]

    return y_predict, e.flatten()


# ***************************************************************************
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
        n_xyz, n_xanes = bootstrap_fn(
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


def bootstrap_test(model_dir, mode, model_mode, xyz_data, xanes_data, ids):
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
                xyz_predict = predict_xyz(xanes_data, model)

                x = xanes_data
                y = xyz_data
                y_predict = xyz_predict

            elif mode == "predict_xanes":
                xanes_predict = predict_xanes(xyz_data, model)

                x = xyz_data
                y = xanes_data
                y_predict = xanes_predict

            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )
            y_predict, e = y_predict_dim(y_predict, ids, model_dir)
            plot.plot_predict(ids, y, y_predict, e, predict_dir, mode)

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            if mode == "predict_xyz":
                recon_xanes, pred_xyz = predict_xyz(xanes_data, model)

                x = xanes_data
                x_recon = recon_xanes
                y = xyz_data
                y_predict = pred_xyz

            elif mode == "predict_xanes":
                recon_xyz, pred_xanes = predict_xanes(xyz_data, model)

                x = xyz_data
                x_recon = recon_xyz
                y = xanes_data
                y_predict = pred_xanes

            print(
                "MSE x to x recon : ",
                mean_squared_error(x, x_recon.detach().numpy()),
            )
            print(
                "MSE y to y pred : ",
                mean_squared_error(y, y_predict.detach().numpy()),
            )

            plot.plot_ae_predict(ids, y, y_predict, x, x_recon, e, predict_dir, mode)

        bootstrap_score.append(mean_squared_error(y, y_predict.detach().numpy()))
    mean_score = torch.mean(torch.tensor(bootstrap_score))
    std_score = torch.std(torch.tensor(bootstrap_score))
    print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}")
