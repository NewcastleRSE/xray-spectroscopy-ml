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
import pickle

import torch
import yaml

from pathlib import Path

from xanesnet.creator import create_predict_scheme, create_descriptor
from xanesnet.data_encoding import encode_predict, encode_predict_gnn
from xanesnet.post_plot import plot_predict, plot_aegan_predict
from xanesnet.post_shap import shap_analysis
from xanesnet.utils import save_predict, save_recon, load_descriptors, load_model_list


def predict_data(config, args, metadata):
    # Load saved metadata from model directory
    metadata_file = Path(f"{args.in_model}/metadata.yaml")
    model_dir = args.in_model

    # Mode consistency check in metadata and args
    meta_mode = metadata["mode"]
    mode = args.mode
    if (meta_mode == "train_xyz" and mode != "predict_xanes") or (
        meta_mode == "train_xanes" and mode != "predict_xyz"
    ):
        raise ValueError(
            f"Inconsistent prediction mode in metadata ({meta_mode}) and args ({args.mode})"
        )
    print(f"Prediction mode: {mode}")
    model_name = metadata["model_type"]

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Enable model evaluation if test data is present
    pred_eval = (mode == "predict_xanes" and config["xanes_path"] is not None) or (
        mode == "predict_xyz" and config["xyz_path"] is not None
    )

    # Encode prediction dataset with saved descriptors
    xyz, xanes, e, index = encode_predict(
        config["xyz_path"], config["xanes_path"], descriptor_list, mode, pred_eval
    )

    # Initialise prediction scheme
    scheme = create_predict_scheme(
        model_name,
        xyz,
        xanes,
        mode,
        index,
        pred_eval,
        metadata["standardscaler"],
        metadata["fourier_transform"],
        metadata["fourier_param"],
    )

    # Predict with loaded models and scheme
    predict_scheme = metadata["scheme"]
    if predict_scheme == "bootstrap":
        if "bootstrap" not in model_dir:
            raise ValueError("Invalid bootstrap directory")

        model_list = load_model_list(Path(model_dir))
        result = scheme.predict_bootstrap(model_list)

    elif predict_scheme == "ensemble":
        if "ensemble" not in model_dir:
            raise ValueError("Invalid ensemble directory")
        model_list = load_model_list(Path(model_dir))
        result = scheme.predict_ensemble(model_list)

    elif predict_scheme == "std":
        model = torch.load(
            Path(model_dir) / "model.pt", map_location=torch.device("cpu")
        )
        result = scheme.predict_std(model)

    else:
        raise ValueError("Unsupported prediction scheme.")

    save_path = "outputs/" + args.in_model
    # Save prediction result
    if config["result_save"]:
        save_predict(save_path, mode, result, index, e)
        if scheme.recon_flag:
            save_recon(save_path, mode, result, index, e)

    # Plot prediction result
    if config["plot_save"]:
        if scheme.recon_flag:
            plot_aegan_predict(save_path, mode, result, index, xyz, xanes)
        else:
            plot_predict(save_path, mode, result, index, xyz, xanes)

    # SHAP analysis
    if config["shap"]:
        nsamples = config["shap_params"]["nsamples"]
        shap_analysis(save_path, mode, model, index, xyz, xanes, nsamples)


def predict_data_gnn(config, args, metadata):
    if args.mode != "predict_xanes":
        raise ValueError(f"Unsupported prediction mode for GNN: {args.mode}")
    print(f"Prediction mode: {args.mode}")
    mode = args.mode

    # Load saved metadata from model directory
    model_dir = Path(args.in_model)

    # Enable model evaluation if test data is present
    pred_eval = config["xanes_path"] is not None

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Encode prediction dataset with saved descriptor
    graph_dataset, index, xanes_data, e = encode_predict_gnn(
        config["xyz_path"],
        config["xanes_path"],
        metadata["node_features"],
        metadata["edge_features"],
        descriptor_list,
        pred_eval,
    )

    # Initialise prediction scheme
    scheme = create_predict_scheme(
        "gnn",
        graph_dataset,
        xanes_data,
        mode,
        index,
        pred_eval,
        metadata["standardscaler"],
        metadata["fourier_transform"],
        metadata["fourier_param"],
    )

    # Predict with loaded models and scheme
    predict_scheme = metadata["scheme"]
    if predict_scheme == "std":
        model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
        result = scheme.predict_std(model)
    # if predict_scheme == "bootstrap":
    #     if "bootstrap" not in str(model_dir):
    #         raise ValueError("Invalid bootstrap directory")
    #
    #     model_list = load_model_list(model_dir)
    #     result = scheme.predict_bootstrap(model_list)
    #
    # elif predict_scheme == "ensemble":
    #     if "ensemble" not in str(model_dir):
    #         raise ValueError("Invalid ensemble directory")
    #     model_list = load_model_list(model_dir)
    #     result = scheme.predict_ensemble(model_list)

    else:
        raise ValueError("Unsupported prediction scheme.")

    save_path = "outputs/" + args.in_model
    # Save prediction result
    if config["result_save"]:
        save_predict(save_path, mode, result, index, e)

    # Plot prediction result
    if config["plot_save"]:
        plot_predict(save_path, mode, result, index, None, xanes_data)

    # SHAP analysis
    # if config["shap"]:
    #     nsamples = config["shap_params"]["nsamples"]
    #     shap_analysis(save_path, mode, model, index, xyz, xanes, nsamples)
