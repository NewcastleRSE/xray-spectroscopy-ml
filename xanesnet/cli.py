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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import sys
from pathlib import Path

import yaml

from argparse import ArgumentParser
from xanesnet.core_learn import train_model, train_model_gnn
from xanesnet.core_predict import predict_data, predict_data_gnn


###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def parse_args(args: list):
    parser = ArgumentParser()

    # available modes:
    # train: train_xanes, train_xyz, train_aegan
    # predict: predict_xyz, predict_xanes, predict_all
    parser.add_argument(
        "--mode",
        type=str,
        help="the mode of the run",
        required=True,
    )

    parser.add_argument(
        "--in_model",
        type=str,
        help="path to pre-trained model directory",
    )
    parser.add_argument(
        "--in_file",
        type=str,
        help="path to .json input file",
        required=True,
    )
    parser.add_argument(
        "--save",
        type=str,
        default="yes",
        help="save result to disk",
    )

    parser.add_argument(
        "--mlflow",
        type=str,
        default="no",
        help="toggle mlflow on and save logs to disk",
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
        if config["model"]["type"] == "gnn":
            train_model_gnn(config, args)
        else:
            train_model(config, args)

    elif "predict" in args.mode:
        metadata_file = Path(f"{args.in_model}/metadata.yaml")
        if os.path.isfile(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)
        else:
            raise ValueError(f"Cannot find metadata file.")

        if metadata["model_type"] == "gnn":
            predict_data_gnn(config, args, metadata)
        else:
            predict_data(config, args, metadata)

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
