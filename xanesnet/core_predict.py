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
from xanesnet.model_utils import make_dir
from xanesnet.utils import save_prediction


def predict_data(config, args):
    # Load saved metadata from model directory
    metadata_file = Path(f"{args.mdl_dir}/metadata.yaml")
    model_dir = Path(args.mdl_dir)

    # Get prediction mode from metafile if present.
    # Otherwise, get the info from args
    if os.path.isfile(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)

        if metadata["mode"] == "train_xyz":
            mode = "predict_xanes"

        elif metadata["mode"] == "train_xanes":
            mode = "predict_xyz"

        elif metadata["mode"] == "train_aegan":
            if config["xyz_path"] is None:
                mode = "predict_xyz"
            elif config["xanes_path"] is None:
                mode = "predict_xanes"
            else:
                mode = "predict_all"

        model_name = metadata["model_mode"]
    else:
        mode = args.mode
        model_name = args.model_mode

    # Load descriptor
    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    # Encode prediction dataset with saved descriptor
    xyz, xanes, index, e = encode_predict(
        config["xyz_path"], config["xanes_path"], descriptor, mode, config["eval"]
    )

    # Initialise prediction scheme
    scheme = create_predict_scheme(
        model_name,
        xyz,
        xanes,
        mode,
        index,
        config["eval"],
        args.fourier_transform,
    )

    # Predict with loaded models and scheme
    if config["bootstrap"]:
        if not str(model_dir).startswith("bootstrap"):
            raise ValueError("Invalid bootstrap directory")

        model_list = load_model_list(model_dir)
        mean, std = scheme.predict(model_list)

    elif config["ensemble"]:
        if not str(model_dir).startswith("ensemble"):
            raise ValueError("Invalid ensemble directory")
        model_list = load_model_list(model_dir)
        mean, std = scheme.predict(model_list)

    else:
        model = torch.load(model_dir / "model.pt", map_location=torch.device("cpu"))
        pred_results = scheme.predict(model)

    # Save prediction result
    if args.save:
        save_path = make_dir()
        # TODO
        pass


def load_model_list(model_dir):
    model_list = []
    n_boot = len(next(os.walk(model_dir))[1])

    for i in range(1, n_boot + 1):
        n_dir = f"{model_dir}/model_{i:03d}/model.pt"
        model = torch.load(n_dir, map_location=torch.device("cpu"))
        model_list.append(model)

    return model_list
