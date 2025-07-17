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

from pathlib import Path

from xanesnet.creator import create_predict_scheme
from xanesnet.encoder import data_predict, data_gnn_predict
from xanesnet.utils.plot import plot_predict, plot_recon_predict
from xanesnet.utils.io import save_predict, load_descriptors, load_models, load_model

# from xanesnet.post_shap import shap_analysis, shap_analysis_gnn


def predict(config, args, metadata):
    """
    Prediction pipeline for non-GNN models.
    """
    model_dir = Path(args.in_model)
    mode = args.mode
    logging.info(f">> Prediction mode: {mode}")

    # Mode consistency check in metadata and args
    _verify_mode(config["xyz_path"], config["xanes_path"], metadata["mode"], mode)

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Enable model evaluation if test data is present
    pred_eval = (config["xyz_path"] is not None) and (config["xanes_path"] is not None)

    # Encode prediction dataset with saved descriptors
    xyz, xanes, e, index = _setup_datasets(
        mode, config, metadata, descriptor_list, pred_eval
    )

    # Initialise prediction scheme
    kwargs = {
        "pred_mode": mode,
        "pred_eval": pred_eval,
        "scaler": metadata["standardscaler"],
        "fourier": metadata["fourier_transform"],
        "fourier_param": metadata["fourier_param"],
    }
    model_type = metadata["model"]["type"]
    scheme = create_predict_scheme(model_type, xyz=xyz, xanes=xanes, **kwargs)

    # Predict with loaded models and scheme
    predict_scheme = metadata["scheme"]
    result, model = _run_prediction(scheme, model_dir, predict_scheme)

    # Set output path
    path = Path("outputs") / args.in_model
    path = Path(str(path).replace("models/", ""))

    # Save prediction result
    if config["result_save"]:
        save_predict(path, mode, result, index, e, scheme.recon_flag)

    # Plot prediction result
    if config["plot_save"]:
        if scheme.recon_flag:
            plot_recon_predict(path, mode, result, index, xyz, xanes)
        else:
            plot_predict(path, mode, result, index, xyz, xanes)

    logging.info("\nPrediction results saved to disk: %s", path.resolve().as_uri())

    # SHAP analysis
    # if config["shap"] and predict_scheme == "std":
    #     xyz_data = scheme.xyz_data
    #     nsamples = config["shap_params"]["nsamples"]
    #     shap_analysis_gnn(path, mode, model, index, xyz_data, xanes_data, nsamples)


def _setup_datasets(mode, config, metadata, descriptor_list, pred_eval):
    model_type = metadata["model"]["type"]

    logging.info(">> Encoding prediction datasets...")
    if model_type.lower() == "gnn":
        if mode != "predict_xanes":
            raise ValueError(f"Unsupported prediction mode for GNN: {mode}")
        xyz, xanes, e, index = data_gnn_predict(
            config["xyz_path"],
            config["xanes_path"],
            metadata["node_features"],
            metadata["edge_features"],
            descriptor_list,
            pred_eval,
        )
    else:
        xyz, xanes, e, index = data_predict(
            config["xyz_path"], config["xanes_path"], descriptor_list, mode, pred_eval
        )

    return xyz, xanes, e, index


def _run_prediction(scheme, model_dir: Path, predict_scheme: str) -> tuple:
    """
    Loads models and runs the prediction based on the specified scheme.
    """
    model = None
    if predict_scheme == "bootstrap":
        if "bootstrap" not in str(model_dir):
            raise ValueError("Invalid bootstrap directory")

        model_list = load_models(model_dir)
        result = scheme.predict_bootstrap(model_list)

    elif predict_scheme == "ensemble":
        if "ensemble" not in str(model_dir):
            raise ValueError("Invalid ensemble directory")
        model_list = load_models(model_dir)
        result = scheme.predict_ensemble(model_list)

    elif predict_scheme == "std" or predict_scheme == "kfold":
        model = load_model(model_dir)
        result = scheme.predict_std(model)

    else:
        raise ValueError("Unsupported prediction scheme.")

    return result, model


def _verify_mode(xyz_path: str, sanes_path: str, meta_mode: str, mode: str):
    """
    Checks for consistency between training mode and prediction mode/data.
    """
    if (meta_mode == "train_xyz" and mode != "predict_xanes") or (
        meta_mode == "train_xanes" and mode != "predict_xyz"
    ):
        raise ValueError(
            f"Inconsistent prediction mode in metadata ({meta_mode}) and args ({mode})"
        )

    if (meta_mode == "train_xyz" and xyz_path is None) or (
        meta_mode == "train_xanes" and sanes_path is None
    ):
        data_type = "xyz" if mode == "train_xyz" else "xanes"
        raise ValueError(f"Cannot find {data_type} prediction dataset.")
