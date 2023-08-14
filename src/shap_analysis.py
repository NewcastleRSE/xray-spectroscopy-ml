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
from model_utils import run_shap_analysis


def shap(
    model_mode, mode, xyz_data, xanes_data, model, predict_dir, ids, shap_nsamples
):
    if model_mode == "mlp" or model_mode == "cnn" or model_mode == "lstm":
        if mode == "predict_xanes":
            data = xyz_data

        elif mode == "predict_xyz":
            data = xanes_data

        data = torch.from_numpy(data).float()

        print(">> Performing SHAP analysis on predicted data...")
        run_shap_analysis(model, predict_dir, data, ids, shap_nsamples)

    elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
        if mode == "predict_xanes":
            # Redefine forward function
            print(">> Performing SHAP analysis on predicted data...")
            model.forward = model.predict
            data = xyz_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="predict",
            )

            print(">> Performing SHAP analysis on reconstructed data...")
            model.forward = model.reconstruct
            data = xyz_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="reconstruct",
            )

        elif mode == "predict_xyz":
            print(">> Performing SHAP analysis on predicted data...")
            model.forward = model.predict
            data = xanes_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="predict",
            )

            print(">> Performing SHAP analysis on reconstructed data...")
            model.foward = model.reconstruct
            data = xanes_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="reconstruct",
            )

    elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":
        if mode == "predict_xanes":
            print(">> Performing SHAP analysis on predicted data...")
            model.forward = model.predict_spectrum
            data = xyz_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="predict",
            )

            print(">> Performing SHAP analysis on reconstructed data...")
            model.forward = model.reconstruct_structure
            data = xyz_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="reconstruct",
            )

        elif mode == "predict_xyz":
            print(">> Performing SHAP analysis on predicted data...")
            model.forward = model.predict_structure
            data = xanes_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="predict",
            )

            print(">> Performing SHAP analysis on reconstructed data...")
            model.forward = model.reconstruct_spectrum
            data = xanes_data
            data = torch.from_numpy(data).float()
            run_shap_analysis(
                model,
                predict_dir,
                data,
                ids,
                shap_nsamples,
                shap_mode="reconstruct",
            )
