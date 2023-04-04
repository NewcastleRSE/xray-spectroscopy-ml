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

import json
import sys
from argparse import ArgumentParser

import yaml

from core_data import train_data
from core_predict import main as predict
from model_utils import json_check
from utils import print_nested_dict

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
		"--mdl_dir",
		type=str,
		help="path to populated model directory during prediction",
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
		type=bool,
		help="SHAP analysis for prediction",
		required=False,
		default=False,
	)
	parser.add_argument(
		"--shap_nsamples",
		type=int,
		help="Number of background samples for SHAP analysis for prediction",
		required=False,
		default=50,
	)

	args = parser.parse_args()

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
    # with open(args.inp_f) as f:
    #     inp = json.load(f)
    with open(args.inp_f, "r") as f:
        inp = yaml.safe_load(f)
    # print_nested_dict(inp, nested_level=1)
    # print("")

    if "train" in args.mode:
        train_data(
            args.mode,
            args.model_mode,
            inp,
            save=args.save,
            fourier_transform=args.fourier_transform,
        )

    elif "predict" in args.mode:
        predict(
            args.mode,
            args.model_mode,
            args.run_shap,
            args.shap_nsamples,
            args.mdl_dir,
            inp,
            fourier_transform=args.fourier_transform,
        )

    else:
        print("wrong mode")


################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


if __name__ == "__main__":
	main(sys.argv[1:])

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
