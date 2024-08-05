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

from xanesnet.creator import create_predict_scheme
from xanesnet.data_descriptor import encode_predict
from xanesnet.post_plot import plot_predict, plot_aegan_predict
from xanesnet.post_shap import shap_analysis
from xanesnet.utils import save_predict, save_recon, load_descriptors, load_model_list


def predict_data(config, args):
    # Load saved metadata from model directory
    metadata_file = Path(f"{args.in_model}/metadata.yaml")
    model_dir = args.in_model

    # Get prediction mode from argument
    if os.path.isfile(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)
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
    else:
        raise ValueError(f"Cannot find metadata file.")

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Enable model evaluation if test data is present
    if (mode == "predict_xanes" and config["xanes_path"] is not None) or (
        mode == "predict_xyz" and config["xyz_path"] is not None
    ):
        pred_eval = True
    else:
        pred_eval = False

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


# def predict_data_gnn(config, args):
#     if args.mode != "predict_xanes":
#         raise ValueError(f"Unsupported prediction mode for GNN: {args.mode}")
#
#     print(f"Prediction mode: {args.mode}")
#
#     # Load saved metadata from model directory
#     metadata_file = Path(f"{args.in_model}/metadata.yaml")
#     model_dir = Path(args.in_model)
#
#     # Get prediction mode from argument
#     if os.path.isfile(metadata_file):
#         with open(metadata_file, "r") as f:
#             metadata = yaml.safe_load(f)
#         model_name = metadata["model_type"]
#     else:
#         raise ValueError(f"Cannot find metadata file.")
#
#     # Load descriptor
#     with open(model_dir / "descriptor.pickle", "rb") as f:
#         descriptor = pickle.load(f)
#
#     # Enable model evaluation if test data is present
#     if (mode == "predict_xanes" and config["xanes_path"] is not None) or (
#         mode == "predict_xyz" and config["xyz_path"] is not None
#     ):
#         pred_eval = True
#     else:
#         pred_eval = False
#
#     # Encode prediction dataset with saved descriptor
#     xyz, xanes, e, index = encode_predict(
#         config["xyz_path"], config["xanes_path"], descriptor, mode, pred_eval
#     )
#
#     # Initialise prediction scheme
#     scheme = create_predict_scheme(
#         model_name,
#         xyz,
#         xanes,
#         mode,
#         index,
#         pred_eval,
#         metadata["standardscaler"],
#         metadata["fourier_transform"],
#         metadata["fourier_param"],
#     )
#
#     # Predict with loaded models and scheme
#     predict_scheme = metadata["scheme"]
#     if predict_scheme == "bootstrap":
#         if "bootstrap" not in str(model_dir):
#             raise ValueError("Invalid bootstrap directory")
#
#         model_list = load_model_list(model_dir)
#         result = scheme.predict_bootstrap(model_list)
#
#     elif predict_scheme == "ensemble":
#         if "ensemble" not in str(model_dir):
#             raise ValueError("Invalid ensemble directory")
#         model_list = load_model_list(model_dir)
#         result = scheme.predict_ensemble(model_list)
#
#     elif predict_scheme == "std":
#         model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
#         result = scheme.predict_std(model)
#
#     else:
#         raise ValueError("Unsupported prediction scheme.")
#
#     save_path = "outputs/" + args.in_model
#     # Save prediction result
#     if config["result_save"]:
#         save_predict(save_path, mode, result, index, e)
#         if scheme.recon_flag:
#             save_recon(save_path, mode, result, index, e)
#
#     # Plot prediction result
#     if config["plot_save"]:
#         if scheme.recon_flag:
#             plot_aegan_predict(save_path, mode, result, index, xyz, xanes)
#         else:
#             plot_predict(save_path, mode, result, index, xyz, xanes)
#
#     # SHAP analysis
#     if config["shap"]:
#         nsamples = config["shap_params"]["nsamples"]
#         shap_analysis(save_path, mode, model, index, xyz, xanes, nsamples)
