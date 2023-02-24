import os
import torch
from sklearn.metrics import mean_squared_error
import model_utils

from predict import predict_xanes
from predict import predict_xyz


def ensemble_predict(ensemble, model_dir, mode, model_mode, xyz_data, xanes_data):
    if ensemble["combine"] == "prediction":
        n_model = len(next(os.walk(model_dir))[1])

        ensemble_preds = []
        ensemble_recons = []
        for i in range(n_model):
            n_dir = f"{model_dir}/model_00{i+1}/model.pt"

            model = torch.load(n_dir, map_location=torch.device("cpu"))
            model.eval()
            print("Loaded model from disk")

            parent_model_dir, predict_dir = model_utils.model_mode_error(
                model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]
            )

            if model_mode == "mlp" or model_mode == "cnn":
                if mode == "predict_xyz":
                    xyz_predict = predict_xyz(xanes_data, model)
                    ensemble_preds.append(xyz_predict)
                    y = xyz_data

                elif mode == "predict_xanes":
                    xanes_predict = predict_xanes(xyz_data, model)
                    ensemble_preds.append(xanes_predict)
                    y = xanes_data

            elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
                if mode == "predict_xyz":
                    xanes_recon, xyz_predict = predict_xyz(xanes_data, model)
                    ensemble_preds.append(xyz_predict)
                    ensemble_recons.append(xanes_recon)

                    x = xanes_data
                    y = xyz_data

                elif mode == "predict_xanes":
                    xyz_recon, xanes_predict = predict_xanes(xyz_data, model)
                    ensemble_preds.append(xanes_predict)
                    ensemble_recons.append(xyz_recon)

                    x = xyz_data
                    y = xanes_data

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
                y_predict = predict_xyz(xanes_data, model)
                y = xyz_data

            elif mode == "predict_xanes":
                y_predict = predict_xyz(xyz_data, model)
                y = xanes_data

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            from model import AutoencoderEnsemble

            model = AutoencoderEnsemble(ensemble_model)
            print("Loaded model from disk")
            if mode == "predict_xyz":
                x_recon, y_predict = predict_xyz(xanes_data, model)

                x = xanes_data
                y = xyz_data

            elif mode == "predict_xanes":
                x_recon, y_predict = predict_xanes(xyz_data, model)

                x = xyz_data
                y = xanes_data
        print(
            "MSE y to y pred : ",
            mean_squared_error(y, y_predict.detach().numpy()),
        )
        if model_mode == "ae_mlp" or model_mode == "ae_cnn":
            print(
                "MSE x to x recon : ",
                mean_squared_error(x, x_recon.detach().numpy()),
            )
