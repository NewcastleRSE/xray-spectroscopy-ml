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

from argparse import ArgumentParser

from ae_learn import main as ae_learn
from ae_predict import main as ae_predict
from utils import print_nested_dict

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################


def parse_args(args: list):

    p = ArgumentParser()

    sub_p = p.add_subparsers(dest="mode")

    learn_p = sub_p.add_parser("train_xyz")
    learn_p.add_argument(
        "inp_f", type=str, help="path to .json input file w/ variable definitions"
    )
    learn_p.add_argument("--model_mode", type=str, help="the model", required=True)
    learn_p.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="toggles model directory creation and population to <off>",
    )

    learn_p = sub_p.add_parser("train_xanes")
    learn_p.add_argument("--model_mode", type=str, help="the model", required=True)
    learn_p.add_argument(
        "inp_f", type=str, help="path to .json input file w/ variable definitions"
    )
    learn_p.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="toggles model directory creation and population to <off>",
    )

    predict_p = sub_p.add_parser("predict_xanes")
    predict_p.add_argument(
        "mdl_dir", type=str, help="path to populated model directory"
    )
    predict_p.add_argument(
        "inp_f", type=str, help="path to .json input file w/ variable definitions"
    )

    predict_p = sub_p.add_parser("predict_xyz")
    predict_p.add_argument(
        "mdl_dir", type=str, help="path to populated model directory"
    )
    predict_p.add_argument(
        "inp_f", type=str, help="path to .json input file w/ variable definitions"
    )

    args = p.parse_args()

    return args


###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################


def main(args: list):

    if len(args) == 0:
        sys.exit()
    else:
        args = parse_args(args)

    if args.mode == "train_xyz":
        print(f">> loading JSON input @ {args.inp_f}\n")
        with open(args.inp_f) as f:
            inp = json.load(f)
        print_nested_dict(inp, nested_level=1)
        print("")
        ae_learn(args.mode, args.model_mode, **inp, save=args.save)

    elif args.mode == "train_xanes":
        print(f">> loading JSON input @ {args.inp_f}\n")
        with open(args.inp_f) as f:
            inp = json.load(f)
        print_nested_dict(inp, nested_level=1)
        print("")
        ae_learn(args.mode, args.model_mode, **inp, save=args.save)

    elif args.mode == "predict_xanes":
        print(f">> loading JSON input @ {args.inp_f}\n")
        with open(args.inp_f) as f:
            inp = json.load(f)
        print_nested_dict(inp, nested_level=1)
        print("")
        ae_predict(args.mode, args.mdl_dir, **inp)

    elif args.mode == "predict_xyz":
        print(f">> loading JSON input @ {args.inp_f}\n")
        with open(args.inp_f) as f:
            inp = json.load(f)
        print_nested_dict(inp, nested_level=1)
        print("")
        ae_predict(args.mode, args.mdl_dir, **inp)


################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == "__main__":
    main(sys.argv[1:])

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
