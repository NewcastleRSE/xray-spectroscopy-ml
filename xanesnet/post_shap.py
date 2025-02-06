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

import random
import shap
import torch
import numpy as np
from pathlib import Path

from xanesnet.utils import mkdir_output


def shap_analysis(path, mode, model, index, xyz, xanes, n_samples):
    save_path = mkdir_output(path, "shap")
    if model.nn_flag:
        if mode == "predict_xyz":
            print(">> Performing SHAP analysis on predicted xyz data...")
            data = xanes
        elif mode == "predict_xanes":
            print(">> Performing SHAP analysis on predicted xanes data...")
            data = xyz

        data = torch.from_numpy(data).float()
        run_shap(model, save_path, data, index, n_samples)

    elif model.ae_flag:
        if mode == "predict_xanes":
            print(">> Performing SHAP analysis on predicted xanes data...")
            data = xyz
            data = torch.from_numpy(data).float()
            model.forward = model.predict
            run_shap(model, save_path, data, index, n_samples, shap_mode="predict")

            print(">> Performing SHAP analysis on reconstructed xanes data...")
            model.forward = model.reconstruct
            run_shap(model, save_path, data, index, n_samples, shap_mode="reconstruct")

        elif mode == "predict_xyz":
            print(">> Performing SHAP analysis on predicted xyz data...")
            data = xanes
            data = torch.from_numpy(data).float()
            model.forward = model.predict
            run_shap(model, save_path, data, index, n_samples, shap_mode="predict")

            print(">> Performing SHAP analysis on reconstructed xyz data...")
            model.forward = model.reconstruct
            run_shap(model, save_path, data, index, n_samples, shap_mode="reconstruct")

    elif model.aegan_flag:
        if mode == "predict_xanes":
            print(">> Performing SHAP analysis on predicted xanes data...")
            data = xyz
            data = torch.from_numpy(data).float()
            model.forward = model.predict_spectrum
            run_shap(model, save_path, data, index, n_samples, shap_mode="predict")

            print(">> Performing SHAP analysis on reconstructed xanes data...")
            model.forward = model.reconstruct_structure
            run_shap(model, save_path, data, index, n_samples, shap_mode="reconstruct")

        elif mode == "predict_xyz":
            print(">> Performing SHAP analysis on predicted xyz data...")
            data = xanes
            data = torch.from_numpy(data).float()
            model.forward = model.predict_spectrum
            run_shap(model, save_path, data, index, n_samples, shap_mode="predict")

            print(">> Performing SHAP analysis on reconstructed xyz data...")
            model.forward = model.reconstruct_spectrum
            run_shap(model, save_path, data, index, n_samples, shap_mode="reconstruct")


def run_shap(model, save_path, data, index, n_samples=100, shap_mode="predict"):
    """
    Get SHAP values for predictions using random sample of data
    as background samples
    """

    shaps_dir = Path(f"{save_path}/shaps-{shap_mode}")
    shaps_dir.mkdir(exist_ok=True)

    n_features = data.shape[1]

    background = data[random.sample(range(data.shape[0]), n_samples)]

    # SHAP analysis
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data)
    shap_values = np.reshape(shap_values, (len(shap_values), data.shape[0], n_features))

    # Print SHAP as a function of features and molecules
    importances = np.mean(np.abs(shap_values), axis=0)
    importances_nonabs = np.mean(shap_values, axis=0)

    overall_imp = np.mean(importances, axis=0)
    energy_imp = np.mean(shap_values, axis=1)

    # SHAP as a function of features and molecules
    for i, id_ in enumerate(index):
        with open(shaps_dir / f"{id_}.shap", "w") as f:
            f.writelines(
                map(
                    "{} {} {}\n".format,
                    np.arange(n_features),
                    importances[i, :],
                    importances_nonabs[i, :],
                )
            )

    # SHAP as a function of features, averaged over all molecules
    with open(shaps_dir / f"overall.shap", "w") as f:
        f.writelines(map("{} {}\n".format, np.arange(n_features), overall_imp))

    # SHAP as a function of features and energy grid points
    energ_dir = shaps_dir / "energy"
    energ_dir.mkdir(exist_ok=True)

    for i in range(shap_values.shape[0]):
        with open(energ_dir / f"energy{i}.shap", "w") as f:
            f.writelines(map("{} {}\n".format, np.arange(n_features), energy_imp[i, :]))
