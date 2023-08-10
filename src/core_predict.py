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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import pickle as pickle
from pathlib import Path

import numpy as np
import torch
import tqdm as tqdm
from sklearn.metrics import mean_squared_error

import data_transform
from inout import load_xanes, load_xyz, save_xanes
from predict import predict_xanes, predict_xyz, y_predict_dim
from utils import linecount, list_filestems
from spectrum.xanes import XANES
from model_utils import make_dir

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(
    mode: str,
    model_mode: str,
    run_shap: bool,
    shap_nsamples: int,
    model_dir: str,
    config,
    fourier_transform: bool = False,
    save: bool = True,
):
    """
    PREDICT. The model state is restored from a model directory containing
    serialised scaling/pipeline objects and the serialised model, .xyz (X)
    data are loaded and transformed, and the model is used to predict XANES
    spectral (Y) data; convolution of the Y data is also possible if
    {conv_params} are provided (see xanesnet/convolute.py).

    Args:
        model_dir (str): The path to a model.[?] directory created by
            the LEARN routine.
        x_path (str): The path to the .xyz (X) data; expects a directory
            containing .xyz files.
    """

    model_dir = Path(model_dir)

    xyz_path = Path(config["x_path"]) if config["x_path"] is not None else None
    xanes_path = Path(
        config["y_path"]) if config["y_path"] is not None else None

    if xyz_path is not None and xanes_path is not None:
        ids = list(set(list_filestems(xyz_path)) &
                   set(list_filestems(xanes_path)))
    elif xyz_path is None:
        ids = list(set(list_filestems(xanes_path)))
    elif xanes_path is None:
        ids = list(set(list_filestems(xyz_path)))

    ids.sort()

    with open(model_dir / "descriptor.pickle", "rb") as f:
        descriptor = pickle.load(f)

    n_samples = len(ids)

    (str(type(descriptor).__name__))

    if xyz_path is not None:
        if str(type(descriptor).__name__) == 'WACSF' or str(type(descriptor).__name__) == 'RDC':
            n_x_features = descriptor.get_len()
        else:
            n_x_features = descriptor.get_number_of_features()
        xyz_data = np.full((n_samples, n_x_features), np.nan)
        print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))

    if xanes_path is not None:
        n_y_features = linecount(xanes_path / f"{ids[0]}.txt") - 2
        xanes_data = np.full((n_samples, n_y_features), np.nan)
        print(">> preallocated {}y{} array for Y data...".format(*xanes_data.shape))

    print(">> ...everything preallocated!\n")

    print(">> loading data into array(s)...")
    if xyz_path is not None:
        if str(type(descriptor).__name__) == 'WACSF' or str(type(descriptor).__name__) == 'RDC':
            for i, id_ in enumerate(tqdm.tqdm(ids)):
                with open(xyz_path / f"{id_}.xyz", "r") as f:
                    atoms = load_xyz(f)
                xyz_data[i, :] = descriptor.transform(atoms)
        elif str(type(descriptor).__name__) == 'MBTR':
            with open(xyz_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
                tmp = descriptor.create(atoms)
            xyz_data[i, :] = tmp
        elif str(type(descriptor).__name__) == 'LMBTR':
            with open(xyz_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
                tmp = descriptor.create(atoms, positions=[0])
            xyz_data[i, :] = tmp
        elif str(type(descriptor).__name__) == 'SOAP':
            with open(xyz_path / f"{id_}.xyz", "r") as f:
                atoms = load_xyz(f)
                tmp = descriptor.create_single(atoms, positions=[0])
            xyz_data[i, :] = tmp
        else:
            print(">> ...This descriptor doesn't exist, try again!!\n")

    if xanes_path is not None:
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(xanes_path / f"{id_}.txt", "r") as f:
                xanes = load_xanes(f)
                e, xanes_data[i, :] = xanes.spectrum
    else:
        e = None

    if xyz_path is None:
        xyz_data = None
    if xanes_path is None:
        xanes_data = None

    print(">> ...loaded!\n")

    if config["bootstrap"]:
        if not str(model_dir).startswith("bootstrap"):
            raise ValueError(
                "Invalid bootstrap directory, please use a bootstrap directory or turn of the bootstrap option in yaml file"
            )
        from bootstrap_fn import bootstrap_predict

        bootstrap_predict(
            model_dir,
            mode,
            model_mode,
            xyz_data,
            xanes_data,
            e,
            ids,
            config["plot_save"],
            fourier_transform,
            config,
        )

    elif config["ensemble"]:
        if not str(model_dir).startswith("ensemble"):
            raise ValueError(
                "Invalid bootstrap directory, please use a bootstrap directory or turn of the bootstrap option in yaml file"
            )
        from ensemble_fn import ensemble_predict

        ensemble_predict(
            config["ensemble_combine"],
            model_dir,
            mode,
            model_mode,
            xyz_data,
            xanes_data,
            e,
            config["plot_save"],
            fourier_transform,
            config,
            ids,
        )

    else:
        model = torch.load(model_dir / "model.pt",
                           map_location=torch.device("cpu"))
        model.eval()
        print("Loaded model from disk")

        # if xyz_path is not None and xanes_path is not None:
        #     from model_utils import model_mode_error

        #     if fourier_transform:
        #         parent_model_dir, predict_dir = model_mode_error(
        #             model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1] * 2
        #         )
        #     else:
        #         parent_model_dir, predict_dir = model_mode_error(
        #             model, mode, model_mode, xyz_data.shape[1], xanes_data.shape[1]
        #         )
        # else:
        

        parent_model_dir, predict_dir = make_dir()

        if model_mode == "mlp" or model_mode == "cnn" or model_mode == "lstm":
            if mode == "predict_xyz":
                if fourier_transform:
                    xanes_data = data_transform.fourier_transform_data(
                        xanes_data)

                xyz_predict = predict_xyz(xanes_data, model)

                x = xanes_data
                y = xyz_data
                y_predict = xyz_predict

            elif mode == "predict_xanes":
                xanes_predict = predict_xanes(xyz_data, model)

                x = xyz_data
                y = xanes_data
                y_predict = xanes_predict

                if fourier_transform:
                    y_predict = data_transform.inverse_fourier_transform_data(
                        y_predict)

            if y is not None:

                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_predict.detach().numpy()),
                )

            y_predict = y_predict_dim(y_predict, ids)
            if y is None:
                # Dummy array for e
                e = np.arange(y_predict.shape[1])

            if config["monte_carlo"]:
                from montecarlo_fn import montecarlo_dropout

                data_compress = {"ids": ids, "y": y,
                                 "y_predict": y_predict, "e": e}
                montecarlo_dropout(
                    model,
                    x,
                    config["mc_iter"],
                    data_compress,
                    predict_dir,
                    mode,
                    config["plot_save"],
                )

            else:
                if save:
                    if mode == "predict_xanes":
                        for id_, y_predict_ in tqdm.tqdm(zip(ids, y_predict)):
                            with open(predict_dir / f"{id_}.txt", "w") as f:
                                save_xanes(
                                    f, XANES(e, y_predict_.detach().numpy()))


                    elif mode == "predict_xyz":
                        for id_, y_predict_ in tqdm.tqdm(zip(ids, y_predict)):
                            with open(predict_dir / f"{id_}.txt", "w") as f:
                                f.write(
                                    "\n".join(
                                        map(str, y_predict_.detach().numpy()))
                                )
                        # for id_, y_ in tqdm.tqdm(zip(ids, y)):
                        #     with open(predict_dir / f"{id_}.wacsf", "w") as f:
                        #         f.write(
                        #             "\n".join(map(str, y_))
                        #         )

                if config["plot_save"]:
                    from plot import plot_predict

                    plot_predict(ids, y, y_predict, predict_dir, mode)

        elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
            if mode == "predict_xyz":
                x = xanes_data
                y = xyz_data

                if fourier_transform:
                    xanes_data = data_transform.fourier_transform_data(
                        xanes_data)

                recon_xanes, pred_xyz = predict_xyz(xanes_data, model)

                x_recon = recon_xanes
                y_predict = pred_xyz

                if fourier_transform:
                    x_recon = data_transform.inverse_fourier_transform_data(
                        x_recon)

            elif mode == "predict_xanes":
                recon_xyz, pred_xanes = predict_xanes(xyz_data, model)

                x = xyz_data
                x_recon = recon_xyz
                y = xanes_data
                y_predict = pred_xanes

                if fourier_transform:
                    y_predict = data_transform.inverse_fourier_transform_data(
                        y_predict)
                    
            if x is not None:
                print(
                    "MSE x to x recon : ",
                    mean_squared_error(x, x_recon.detach().numpy()),
                )

            if y is not None:
                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_predict.detach().numpy()),
                )

            if y is None: 
                # Dummy array for e
                e = np.arange(y_predict.shape[1])


            if config["monte_carlo"]:
                from montecarlo_fn import montecarlo_dropout_ae

                data_compress = {
                    "ids": ids,
                    "y": y,
                    "y_predict": y_predict,
                    "e": e,
                    "x": x,
                    "x_recon": x_recon,
                }
                montecarlo_dropout_ae(
                    model,
                    x,
                    config["mc_iter"],
                    data_compress,
                    predict_dir,
                    mode,
                    config["plot_save"],
                )

            else:
                y_predict = y_predict_dim(y_predict, ids)



                if save:
                    if mode == "predict_xanes":
                        for id_, y_predict_ in tqdm.tqdm(zip(ids, y_predict)):
                            with open(predict_dir / f"{id_}.txt", "w") as f:
                                save_xanes(
                                    f, XANES(e, y_predict_.detach().numpy()))

                    elif mode == "predict_xyz":
                        for id_, y_predict_ in tqdm.tqdm(zip(ids, y_predict)):
                            with open(predict_dir / f"{id_}.txt", "w") as f:
                                f.write(
                                    "\n".join(
                                        map(str, y_predict_.detach().numpy()))
                                )
                        # for id_, y_ in tqdm.tqdm(zip(ids, y)):
                        #     with open(predict_dir / f"{id_}.wacsf", "w") as f:
                        #         f.write(
                        #             "\n".join(map(str, y_))
                        #         )

                if config["plot_save"]:
                    from plot import plot_ae_predict

                    plot_ae_predict(ids, y, y_predict, x,
                                    x_recon, e, predict_dir, mode)

        elif model_mode == "aegan_mlp" or model_mode == "aegan_cnn":

            x = xyz_data
            y = xanes_data

            import aegan_predict

            x_recon, y_pred, y_recon, x_pred = aegan_predict.predict_aegan(x, y, model, mode, fourier_transform)

            if x is not None and x_recon is not None:
                print(
                    "MSE x to x recon : ",
                    mean_squared_error(x, x_recon.detach().numpy()),
                )
            if x is not None and x_pred is not None:
                print(
                    "MSE x to x pred : ",
                    mean_squared_error(x, x_pred.detach().numpy()),
                )

            if y is not None and y_recon is not None:
                print(
                    "MSE y to y recon : ",
                    mean_squared_error(y, y_recon.detach().numpy()),
                )

            if y is not None and y_pred is not None:
                print(
                    "MSE y to y pred : ",
                    mean_squared_error(y, y_pred.detach().numpy()),
                )

            if y is None: 
                # Dummy array for e
                e = np.arange(y_pred.shape[1])

            if config["monte_carlo"]:
                from montecarlo_fn import montecarlo_dropout_aegan

                montecarlo_dropout_aegan(model, x, y, config["mc_iter"])

            else:
                if save:
                    if mode == "predict_xanes":
                        
                        for id_, y_pred_ in tqdm.tqdm(zip(ids, y_pred)):
                            with open(predict_dir / f"predict-{id_}.txt", "w") as f:
                                save_xanes(
                                    f, XANES(e, y_pred_.detach().numpy()))
                                
                        for id_, x_recon_ in tqdm.tqdm(zip(ids, x_recon)):
                            with open(predict_dir / f"recon-{id_}.txt", "w") as f:
                                f.write(
                                    "\n".join(
                                        map(str, x_recon_.detach().numpy()))
                                )

                    elif mode == "predict_xyz":
                        
                        for id_, x_pred_ in tqdm.tqdm(zip(ids, x_pred)):
                            with open(predict_dir / f"predict-{id_}.txt", "w") as f:
                                f.write(
                                    "\n".join(
                                        map(str, x_pred_.detach().numpy()))
                                )
                        for id_, y_recon_ in tqdm.tqdm(zip(ids, y_recon)):
                            with open(predict_dir / f"recon-{id_}.txt", "w") as f:
                                save_xanes(
                                    f, XANES(e, y_recon_.detach().numpy()))
                
                    elif mode == "predict_all":

                        for id_, y_pred_ in tqdm.tqdm(zip(ids, y_pred)):
                            with open(predict_dir / f"predict-{id_}.txt", "w") as f:
                                save_xanes(
                                    f, XANES(e, y_pred_.detach().numpy()))
                                
                        for id_, x_recon_ in tqdm.tqdm(zip(ids, x_recon)):
                            with open(predict_dir / f"recon-{id_}.txt", "w") as f:
                                f.write(
                                    "\n".join(
                                        map(str, x_recon_.detach().numpy()))
                                )
                        for id_, x_pred_ in tqdm.tqdm(zip(ids, x_pred)):
                            with open(predict_dir / f"predict-{id_}.txt", "w") as f:
                                f.write(
                                    "\n".join(
                                        map(str, x_pred_.detach().numpy()))
                                )
                        for id_, y_recon_ in tqdm.tqdm(zip(ids, y_recon)):
                            with open(predict_dir / f"recon-{id_}.txt", "w") as f:
                                save_xanes(
                                    f, XANES(e, y_recon_.detach().numpy()))

                if config["plot_save"]:
                    from plot import plot_aegan_predict

                    plot_aegan_predict(ids, x, y, x_recon, y_recon, x_pred, y_pred, predict_dir, mode)


            # # Convert to float
            # if config["x_path"] is not None and config["y_path"] is not None:
            #     x = torch.tensor(xyz_data).float()
            #     y = torch.tensor(xanes_data).float()
            # elif config["x_path"] is not None and config["y_path"] is None:
            #     x = torch.tensor(xyz_data).float()
            #     y = None
            #     e = None
            # elif config["y_path"] is not None and config["x_path"] is None:
            #     y = torch.tensor(xanes_data).float()
            #     x = None

           # if config["monte_carlo"]:
            #     from montecarlo_fn import montecarlo_dropout_aegan

            #     montecarlo_dropout_aegan(model, x, y, config["mc_iter"]) 
            # else:
            #     import aegan_predict

            #     aegan_predict.main(
            #         config,
            #         x,
            #         y,
            #         model,
            #         fourier_transform,
            #         model_dir,
            #         predict_dir,
            #         ids,
            #         parent_model_dir,
            #         e
            #     )

        if run_shap:
            from shap_analysis import shap

            shap(
                model_mode,
                mode,
                xyz_data,
                xanes_data,
                model,
                predict_dir,
                ids,
                shap_nsamples,
            )
