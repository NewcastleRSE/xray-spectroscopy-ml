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

import sys
import yaml

from argparse import ArgumentParser
from tblite.interface import Calculator
from xanesnet.core_learn import train_model
from xanesnet.core_predict import predict_data


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
        "--in_model",
        type=str,
        help="path to populated model directory during prediction",
    )
    parser.add_argument(
        "--in_file",
        type=str,
        help="path to .json input file w/ variable definitions",
        required=True,
    )
    parser.add_argument(
        "--save",
        type=str,
        default="yes",
        help="toggles model directory creation and population to <on>",
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

    print(f">> loading JSON input @ {args.in_file}\n")

    with open(args.in_file, "r") as f:
        config = yaml.safe_load(f)

    if "train" in args.mode:
        train_model(config, args)

    elif "predict" in args.mode:
        predict_data(config, args)

    else:
        print(">>> Incorrect mode. Please try again.")


################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


if __name__ == "__main__":
    main(sys.argv[1:])

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
