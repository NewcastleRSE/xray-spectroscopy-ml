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

import torch

from pathlib import Path
from xanesnet.creator import create_predict_scheme
from xanesnet.data_encoding import data_predict, data_gnn_predict
from xanesnet.post_plot import plot_predict, plot_recon_predict
from xanesnet.post_shap import shap_analysis, shap_analysis_gnn
from xanesnet.utils import save_predict, load_descriptors, load_models


def predict_data(config, args, metadata):
    model_dir = Path(args.in_model)

    # Mode consistency check in metadata and args
    meta_mode = metadata["mode"]
    mode = args.mode
    print(f"Prediction mode: {mode}")
    consistency_check(config, meta_mode, mode)

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Enable model evaluation if test data is present
    pred_eval = (config["xyz_path"] is not None) and (config["xanes_path"] is not None)

    # Encode prediction dataset with saved descriptors
    xyz, xanes, e, index = data_predict(
        config["xyz_path"], config["xanes_path"], descriptor_list, mode, pred_eval
    )

    # Initialise prediction scheme
    scheme = create_predict_scheme(
        metadata["model_type"],
        xyz,
        xanes,
        mode,
        pred_eval,
        metadata["standardscaler"],
        metadata["fourier_transform"],
        metadata["fourier_param"],
    )

    # Predict with loaded models and scheme
    predict_scheme = metadata["scheme"]
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
        model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
        result = scheme.predict_std(model)

    else:
        raise ValueError("Unsupported prediction scheme.")

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

    # SHAP analysis
    if config["shap"] and predict_scheme == "std":
        nsamples = config["shap_params"]["nsamples"]
        shap_analysis(path, mode, model, index, xyz, xanes, nsamples)


def predict_data_gnn(config, args, metadata):
    model_dir = Path(args.in_model)
    if args.mode != "predict_xanes":
        raise ValueError(f"Unsupported prediction mode for GNN: {args.mode}")
    mode = args.mode
    print(f"Prediction mode: {mode}")

    # Enable model evaluation if test data is present
    pred_eval = config["xanes_path"] is not None

    # Load descriptor list
    descriptor_list = load_descriptors(model_dir)

    # Encode prediction dataset with saved descriptor
    graph_dataset, index, xanes_data, e = data_gnn_predict(
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
        pred_eval,
        metadata["standardscaler"],
        metadata["fourier_transform"],
        metadata["fourier_param"],
    )

    # Predict with loaded models and scheme
    predict_scheme = metadata["scheme"]
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
        model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
        result = scheme.predict_std(model)

    else:
        raise ValueError("Unsupported prediction scheme.")

    path = Path("outputs") / args.in_model
    path = Path(str(path).replace("models/", ""))

    # Save prediction result
    if config["result_save"]:
        save_predict(path, mode, result, index, e, False)

    # Plot prediction result
    if config["plot_save"]:
        plot_predict(path, mode, result, index, None, xanes_data)

    # SHAP analysis
    if config["shap"] and predict_scheme == "std":
        xyz_data = scheme.xyz_data
        nsamples = config["shap_params"]["nsamples"]
        shap_analysis_gnn(path, mode, model, index, xyz_data, xanes_data, nsamples)


def consistency_check(config, meta_mode, mode):
    if (meta_mode == "train_xyz" and mode != "predict_xanes") or (
        meta_mode == "train_xanes" and mode != "predict_xyz"
    ):
        raise ValueError(
            f"Inconsistent prediction mode in metadata ({meta_mode}) and args ({mode})"
        )

    if (meta_mode == "train_xyz" and config["xyz_path"] is None) or (
        meta_mode == "train_xanes" and config["xanes_path"] is None
    ):
        raise ValueError(
            f"Missing {'xyz' if meta_mode == 'train_xyz' else 'xanes'} data"
        )
