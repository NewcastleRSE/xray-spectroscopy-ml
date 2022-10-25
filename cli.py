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
# import importlib.resources

from argparse import ArgumentParser

from core_learn import main as learn
from core_predict import main as predict
from utils import print_nested_dict

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args(args: list):

    p = ArgumentParser()
    
    sub_p = p.add_subparsers(dest = 'mode')

    learn_p = sub_p.add_parser('learn')
    learn_p.add_argument('inp_f', type = str, 
        help = 'path to .json input file w/ variable definitions')
    learn_p.add_argument('--no-save', dest = 'save', action = 'store_false',
        help = 'toggles model directory creation and population to <off>')

    predict_p = sub_p.add_parser('predict')
    predict_p.add_argument('mdl_dir', type = str, 
        help = 'path to populated model directory')
    predict_p.add_argument('xyz_dir', type = str, 
        help = 'path to .xyz input directory for prediction')
    predict_p.add_argument('xanes_dir', type = str, 
        help = 'path to xanes directory for prediction')
    
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
        

    if args.mode == 'learn':
        print(f'>> loading JSON input @ {args.inp_f}\n')
        with open(args.inp_f) as f:
            inp = json.load(f)
        print_nested_dict(inp, nested_level = 1)
        print('')
        learn(**inp, save = args.save)
        print("done")

    if args.mode == 'predict':
        predict(args.mdl_dir, args.xyz_dir, args.xanes_dir)
    

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main(sys.argv[1:])
    print("done")

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################