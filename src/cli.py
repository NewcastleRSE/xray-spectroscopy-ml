"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

import sys
import json
import importlib.resources

from argparse import ArgumentParser

from core_data import train_data

# from core_learn import main as learn
from core_predict import main as predict

from core_eval import main as eval_model
from utils import print_nested_dict
from model_utils import json_check

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def parse_args(args: list):
    parser = ArgumentParser()

    # mode
    # train_xanes, train_xyz, train_aegan, predict_xyz,
    # predict_xanes, predict_aegan, predict_aegan_xanes, predict_aegan_xyz,
    # eval_pred_xanes, eval_pred_xyz, eval_recon_xanes, eval_recon_xyz
    parser.add_argument(
        "--mode",
        type=str,
        help="the mode of the run",
        required=True,
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        help="the model to use to train or to predict",
        required=True,
    )
    parser.add_argument(
        "--mdl_dir", type=str, help="path to populated model directory during prediction"
    )
    parser.add_argument(
        "--inp_f",
        type=str,
        help="path to .json input file w/ variable definitions",
        required=True,
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="toggles model directory creation and population to <off>",
    )
    parser.add_argument(
        "--fourier_transform",
        action="store_true",
        help="Train using Fourier transformed xanes spectra or Predict using model trained on Fourier transformed xanes spectra",
    )

    parser.add_argument(
        "--run_shap",
        type=bool, help="SHAP analysis for prediction", required=False, default=False,
    )
    parser.add_argument(
        "--shap_nsamples",
        type=int,
        help="Number of background samples for SHAP analysis for prediction",
        required=False,
        default=50,
    )

    args = parser.parse_args()

    # p=ArgumentParser()

    # sub_p=p.add_subparsers(dest = "mode")

    # learn_p=sub_p.add_parser("train_xyz")
    # learn_p.add_argument(
    #     "inp_f", type = str, help = "path to .json input file w/ variable definitions"
    # )
    # learn_p.add_argument("--model_mode", type = str,
    #                      help = "the model", required = True)
    # learn_p.add_argument(
    #     "--no-save",
    #     dest="save",
    #     action="store_false",
    #     help="toggles model directory creation and population to <off>",
    # )
    # learn_p.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Train using Fourier transformed xanes spectra",
    # )

    # learn_p = sub_p.add_parser("train_xanes")
    # learn_p.add_argument("--model_mode", type=str,
    #                      help="the model", required=True)
    # learn_p.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # learn_p.add_argument(
    #     "--no-save",
    #     dest="save",
    #     action="store_false",
    #     help="toggles model directory creation and population to <off>",
    # )
    # learn_p.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Train using Fourier transformed xanes spectra",
    # )

    # learn_p = sub_p.add_parser("train_aegan")
    # learn_p.add_argument("--model_mode", type=str,
    #                      help="the model", required=True)
    # learn_p.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # learn_p.add_argument(
    #     "--no-save",
    #     dest="save",
    #     action="store_false",
    #     help="toggles model directory creation and population to <off>",
    # )
    # learn_p.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Train using Fourier transformed xanes spectra",
    # )

    # predict_p = sub_p.add_parser("predict_xanes")
    # predict_p.add_argument("--model_mode", type=str,
    #                        help="the model", required=True)
    # # shap arguments
    # predict_p.add_argument(
    #     "--run_shap", type=bool, help="SHAP analysis", required=False, default=False
    # )
    # predict_p.add_argument(
    #     "--shap_nsamples",
    #     type=int,
    #     help="Number of background samples for SHAP analysis",
    #     required=False,
    #     default=50,
    # )
    # predict_p.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # predict_p.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # predict_p.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Predict using model trained on Fourier transformed xanes spectra",
    # )

    # predict_p = sub_p.add_parser("predict_xyz")
    # predict_p.add_argument("--model_mode", type=str,
    #                        help="the model", required=True)
    # # shap arguments
    # predict_p.add_argument(
    #     "--run_shap", type=bool, help="SHAP analysis", required=False, default=False
    # )
    # predict_p.add_argument(
    #     "--shap_nsamples",
    #     type=int,
    #     help="Number of background samples for SHAP analysis",
    #     required=False,
    #     default=50,
    # )
    # predict_p.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # predict_p.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # predict_p.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Predict using model trained on Fourier transformed xanes spectra",
    # )

    # # Parser for structural and spectral inputs
    # predict_p = sub_p.add_parser("predict_aegan")
    # predict_p.add_argument("--model_mode", type=str,
    #                        help="the model", required=True)
    # # shap arguments
    # predict_p.add_argument(
    #     "--run_shap", type=bool, help="SHAP analysis", required=False, default=False
    # )
    # predict_p.add_argument(
    #     "--shap_nsamples",
    #     type=int,
    #     help="Number of background samples for SHAP analysis",
    #     required=False,
    #     default=50,
    # )
    # predict_p.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # predict_p.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # predict_p.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Predict using model trained on Fourier transformed xanes spectra",
    # )

    # # Parser for structual inputs only
    # predict_p_xyz = sub_p.add_parser("predict_aegan_xanes")
    # predict_p_xyz.add_argument(
    #     "--model_mode", type=str, help="the model", required=True
    # )
    # # shap arguments
    # predict_p_xyz.add_argument(
    #     "--run_shap", type=bool, help="SHAP analysis", required=False, default=False
    # )
    # predict_p_xyz.add_argument(
    #     "--shap_nsamples",
    #     type=int,
    #     help="Number of background samples for SHAP analysis",
    #     required=False,
    #     default=50,
    # )
    # predict_p_xyz.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # predict_p_xyz.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # predict_p_xyz.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Predict using model trained on Fourier transformed xanes spectra",
    # )

    # predict_p_xanes = sub_p.add_parser("predict_aegan_xyz")
    # predict_p_xanes.add_argument(
    #     "--model_mode", type=str, help="the model", required=True
    # )
    # # shap arguments
    # predict_p_xanes.add_argument(
    #     "--run_shap", type=bool, help="SHAP analysis", required=False, default=False
    # )
    # predict_p_xanes.add_argument(
    #     "--shap_nsamples",
    #     type=int,
    #     help="Number of background samples for SHAP analysis",
    #     required=False,
    #     default=50,
    # )
    # predict_p_xanes.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # predict_p_xanes.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )
    # predict_p_xanes.add_argument(
    #     "--fourier_transform",
    #     action="store_true",
    #     help="Predict using model trained on Fourier transformed xanes spectra",
    # )

    # eval_p_pred_xanes = sub_p.add_parser("eval_pred_xanes")
    # eval_p_pred_xanes.add_argument(
    #     "--model_mode", type=str, help="the model", required=True
    # )
    # eval_p_pred_xanes.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # eval_p_pred_xanes.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )

    # eval_p_pred_xyz = sub_p.add_parser("eval_pred_xyz")
    # eval_p_pred_xyz.add_argument(
    #     "--model_mode", type=str, help="the model", required=True
    # )
    # eval_p_pred_xyz.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # eval_p_pred_xyz.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )

    # eval_p_recon_xanes = sub_p.add_parser("eval_recon_xanes")
    # eval_p_recon_xanes.add_argument(
    #     "--model_mode", type=str, help="the model", required=True
    # )
    # eval_p_recon_xanes.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # eval_p_recon_xanes.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )

    # eval_p_recon_xyz = sub_p.add_parser("eval_recon_xyz")
    # eval_p_recon_xyz.add_argument(
    #     "--model_mode", type=str, help="the model", required=True
    # )
    # eval_p_recon_xyz.add_argument(
    #     "mdl_dir", type=str, help="path to populated model directory"
    # )
    # eval_p_recon_xyz.add_argument(
    #     "inp_f", type=str, help="path to .json input file w/ variable definitions"
    # )

    # args = p.parse_args()

    return args


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list):
    if len(args) == 0:
        sys.exit()
    else:
        args = parse_args(args)

    print(f">> loading JSON input @ {args.inp_f}\n")
    with open(args.inp_f) as f:
        inp = json.load(f)
    print_nested_dict(inp, nested_level=1)
    print("")

    if "train" in args.mode:
        train_data(
            args.mode,
            args.model_mode,
            **inp,
            save=args.save,
            fourier_transform=args.fourier_transform,
        )

    # elif args.mode == "train_xanes":
    #     json_check(inp)
    #     train_data(
    #         args.mode,
    #         args.model_mode,
    #         **inp,
    #         save=args.save,
    #         fourier_transform=args.fourier_transform,
    #     )

    # elif args.mode == "train_aegan":
    #     train_data(
    #         args.mode,
    #         args.model_mode,
    #         **inp,
    #         save=args.save,
    #         fourier_transform=args.fourier_transform,
    #     )

    elif "predict" in args.mode:
        predict(
            args.mode,
            args.model_mode,
            args.run_shap,
            args.shap_nsamples,
            args.mdl_dir,
            **inp,
            fourier_transform=args.fourier_transform,
        )

    if "eval" in args.mode:
        eval_model(
            args.mode,
            args.mdl_dir,
            args.run_shap,
            args.shap_nsamples,
            args.model_mode,
            **inp,
        )


################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == "__main__":
    main(sys.argv[1:])

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
