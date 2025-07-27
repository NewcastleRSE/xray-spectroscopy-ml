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

import logging
from enum import Enum

from pathlib import Path

import numpy as np

from xanesnet.creator import create_predict_scheme, create_dataset
from xanesnet.utils.plot import plot_predict, plot_recon_predict
from xanesnet.utils.io import (
    save_predict_result,
    load_descriptors_from_local,
    load_models_from_local,
    load_model_from_local,
)

# from xanesnet.post_shap import shap_analysis, shap_analysis_gnn


def predict(config, args, metadata):
    """
    Prediction pipeline for non-GNN models.
    """
    model_dir = Path(args.in_model)
    xyz_path = config["dataset"].get("xyz_path", None)
    xanes_path = config["dataset"].get("xanes_path", None)

    mode = args.mode
    _verify_mode(xyz_path, xanes_path, metadata, mode)
    logging.info(f">> Prediction mode: {mode}")

    # Load descriptor list
    descriptor_list = load_descriptors_from_local(model_dir)

    # Enable model evaluation if test data is present
    pred_eval = xyz_path is not None and xanes_path is not None

    # Load, encode, and preprocess data
    dataset = _setup_datasets(config, metadata, descriptor_list)

    # Setup prediction scheme
    scheme = _setup_scheme(metadata, mode, pred_eval, dataset)

    # Predict with loaded models and scheme
    saved_train_scheme = metadata["scheme"]
    result, model = _run_prediction(scheme, model_dir, saved_train_scheme)

    # Set output path
    path = Path("outputs") / args.in_model
    path = Path(str(path).replace("models/", ""))

    # Save prediction result
    if config["result_save"]:
        save_predict_result(
            path, mode, result, dataset.index, dataset.e_data, scheme.recon_flag
        )

    # Plot prediction result
    if config["plot_save"]:
        if scheme.recon_flag:
            plot_recon_predict(
                path, mode, result, dataset.index, dataset.xyz_data, dataset.xanes_data
            )
        else:
            plot_predict(
                path, mode, result, dataset.index, dataset.xyz_data, dataset.xanes_data
            )

    logging.info("\nPrediction results saved to disk: %s", path.resolve().as_uri())

    # SHAP analysis
    # if config["shap"] and predict_scheme == "std":
    #     xyz_data = scheme.xyz_data
    #     nsamples = config["shap_params"]["nsamples"]
    #     shap_analysis_gnn(path, mode, model, index, xyz_data, xanes_data, nsamples)


def _setup_datasets(config, metadata, descriptor_list):
    logging.info(">> Initialising prediction datasets...")
    dataset_type = metadata["dataset"]["type"]
    root_path = config["dataset"].get("root_path")
    xyz_path = config["dataset"].get("xyz_path", None)
    xanes_path = config["dataset"].get("xanes_path", None)

    # Pack kwargs
    kwargs = {
        "root": root_path,
        "xyz_path": xyz_path,
        "xanes_path": xanes_path,
        "descriptors": descriptor_list,
        "shuffle": False,
        **metadata["dataset"]["params"],
    }

    dataset = create_dataset(dataset_type, **kwargs)
    return dataset


def _setup_scheme(metadata, mode, pred_eval, dataset):
    kwargs = {
        "pred_mode": mode,
        "pred_eval": pred_eval,
        "scaler": metadata["standardscaler"],
        "fourier": metadata["dataset"]["params"]["fourier"],
        "fourier_param": metadata["dataset"]["params"]["fourier_concat"],
    }

    model_type = metadata["model"]["type"]
    if model_type.lower() == "gnn":
        xanes_data = None
        if pred_eval:
            xanes_data = [graph.y.numpy() for graph in dataset]
            xanes_data = np.array(xanes_data)
        xyz, xanes = dataset, xanes_data

        logging.info(
            f">> Graph dataset (samples: {len(xyz)}, "
            f"node features: {dataset[0].x.shape[1]}, "
            f"edge features: {dataset[0].edge_attr.shape[1]}, "
            f"graph features: {dataset[0].graph_attr.shape[0]})"
        )
    else:
        xyz, xanes = dataset.xyz_data, dataset.xanes_data
        if xyz is not None:
            logging.info(f">> xyz (samples: {xyz.shape[0]}, features: {xyz.shape[1]})")
        if xanes is not None:
            logging.info(
                f">> xanes (samples: {xanes.shape[0]}, features: {xanes.shape[1]})"
            )

    scheme = create_predict_scheme(model_type, xyz=xyz, xanes=xanes, **kwargs)
    return scheme


def _run_prediction(scheme, model_dir: Path, predict_scheme: str) -> tuple:
    """
    Loads models and runs the prediction based on the specified scheme.
    """
    model = None
    if predict_scheme == "bootstrap":
        if "bootstrap" not in str(model_dir):
            raise ValueError("Invalid bootstrap directory")

        model_list = load_models_from_local(model_dir)
        result = scheme.predict_bootstrap(model_list)

    elif predict_scheme == "ensemble":
        if "ensemble" not in str(model_dir):
            raise ValueError("Invalid ensemble directory")
        model_list = load_models_from_local(model_dir)
        result = scheme.predict_ensemble(model_list)

    elif predict_scheme == "std" or predict_scheme == "kfold":
        model = load_model_from_local(model_dir)
        result = scheme.predict_std(model)

    else:
        raise ValueError("Unsupported prediction scheme.")

    return result, model


def _verify_mode(xyz_path, xanes_path, metadata, mode):
    """
    Checks for consistency between training mode and prediction mode/data.
    """
    meta_mode = metadata["mode"]

    if (meta_mode == "train_xyz" and mode != "predict_xanes") or (
        meta_mode == "train_xanes" and mode != "predict_xyz"
    ):
        raise ValueError(
            f"Inconsistent prediction mode in metadata ({meta_mode}) and args ({mode})"
        )

    if (meta_mode == "train_xyz" and xyz_path is None) or (
        meta_mode == "train_xanes" and xanes_path is None
    ):
        data_type = "xyz" if mode == "train_xyz" else "xanes"
        raise ValueError(f"Cannot find {data_type} prediction dataset.")
