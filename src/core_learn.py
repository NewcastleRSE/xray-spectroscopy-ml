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
import tqdm as tqdm

from torchinfo import summary

from utils import print_cross_validation_scores

from learn import train
from ae_learn import train as ae_train
from aegan_learn import train_aegan as aegan_train

from kfold_fn import kfold_train
from kfold_fn import kfold_ae_train
from kfold_fn import kfold_aegan_train


def train_xyz(
	xyz,
	xanes,
	exp_name,
	model_mode,
	hyperparams,
	epochs,
	kfold,
	kfold_params,
	rng,
	weight_seed,
	lr_scheduler,
	model_eval,
	load_guess,
	loadguess_params,
	optuna_params,
):
	print("training xyz structure")

	if model_mode == "mlp" or model_mode == "cnn" or model_mode == "lstm":
		if kfold:
			x = xyz
			y = xanes
			result, model = kfold_train(
				x,
				y,
				kfold_params,
				rng,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)
			print_cross_validation_scores(result, model_mode)
		else:
			if optuna_params["tune"]:
				from optuna_learn import main as learn_optparams

				print(">> Finding optimal hyperparams...")
				opt_trial, opt_score = learn_optparams(optuna_params,
								xyz,
								xanes,
								exp_name,
								model_mode,
								hyperparams,
								epochs,
								weight_seed,
								lr_scheduler,
								model_eval,
							)
				hyperparams.update(opt_trial.params)

			print(">> fitting neural net...")
			model, score = train(
				xyz,
				xanes,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)

	elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
		if kfold:
			x = xyz
			y = xanes
			result, model = kfold_ae_train(
				x,
				y,
				kfold_params,
				rng,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)
			print_cross_validation_scores(result, model_mode)
		else:

			if optuna_params["tune"]:
				from optuna_learn import main as learn_optparams

				print(">> Finding optimal hyperparams...")
				opt_trial, opt_score = learn_optparams(optuna_params,
								xyz,
								xanes,
								exp_name,
								model_mode,
								hyperparams,
								epochs,
								weight_seed,
								lr_scheduler,
								model_eval,
							)
				hyperparams.update(opt_trial.params)

			print(">> fitting neural net...")
			model, score = ae_train(
				xyz,
				xanes,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)

	summary(model, (1, xyz.shape[1]))
	return model


def train_xanes(
	xyz,
	xanes,
	exp_name,
	model_mode,
	hyperparams,
	epochs,
	kfold,
	kfold_params,
	rng,
	weight_seed,
	lr_scheduler,
	model_eval,
	load_guess,
	loadguess_params,
	optuna_params,
):
	print("training xanes spectrum")

	if model_mode == "mlp" or model_mode == "cnn" or model_mode == "lstm":
		if kfold:
			x = xanes
			y = xyz
			result, model = kfold_train(
				x,
				y,
				kfold_params,
				rng,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)
			print_cross_validation_scores(result, model_mode)
		else:
			if optuna_params["tune"]:
				from optuna_learn import main as learn_optparams

				print(">> Finding optimal hyperparams...")
				opt_trial, opt_score = learn_optparams(optuna_params,
								xyz,
								xanes,
								exp_name,
								model_mode,
								hyperparams,
								epochs,
								weight_seed,
								lr_scheduler,
								model_eval,
							)
				hyperparams.update(opt_trial.params)

			print(">> fitting neural net...")
			model, score = train(
				xanes,
				xyz,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)

	elif model_mode == "ae_mlp" or model_mode == "ae_cnn":
		if kfold:
			x = xanes
			y = xyz
			result, model = kfold_ae_train(
				x,
				y,
				kfold_params,
				rng,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)
			print_cross_validation_scores(result, model_mode)

		else:
			if optuna_params["tune"]:
				from optuna_learn import main as learn_optparams

				print(">> Finding optimal hyperparams...")
				opt_trial, opt_score = learn_optparams(optuna_params,
								xyz,
								xanes,
								exp_name,
								model_mode,
								hyperparams,
								epochs,
								weight_seed,
								lr_scheduler,
								model_eval,
							)
				hyperparams.update(opt_trial.params)

			print(">> fitting neural net...")
			model, score = ae_train(
				xanes,
				xyz,
				exp_name,
				model_mode,
				hyperparams,
				epochs,
				weight_seed,
				lr_scheduler,
				model_eval,
				load_guess,
				loadguess_params,
			)

	summary(model, (1, xanes.shape[1]))
	return model


def train_aegan(
	xyz,
	xanes,
	exp_name,
	model_mode,
	hyperparams,
	epochs,
	kfold,
	kfold_params,
	rng,
	weight_seed,
	lr_scheduler,
	model_eval,
	load_guess,
	loadguess_params,
	optuna_params,
):
	if kfold:
		result, model = kfold_aegan_train(
			xyz,
			xanes,
			kfold_params,
			rng,
			exp_name,
			model_mode,
			hyperparams,
			epochs,
			weight_seed,
			lr_scheduler,
			model_eval,
	  		load_guess,
	  		loadguess_params,
		)
		print_cross_validation_scores(result, model_mode)

	else:
		if optuna_params["tune"]:
			from optuna_learn import main as learn_optparams

			print(">> Finding optimal hyperparams...")
			opt_trial, opt_score = learn_optparams(optuna_params,
							xyz,
							xanes,
							exp_name,
							model_mode,
							hyperparams,
							epochs,
							weight_seed,
							lr_scheduler,
							model_eval,
						)
			hyperparams.update(opt_trial.params)

		print(">> fitting neural net...")
		model, score = aegan_train(
			xyz,
			xanes,
			exp_name,
			hyperparams,
			epochs,
			weight_seed,
			lr_scheduler,
			model_eval,
	  		load_guess,
	  		loadguess_params,
		)
	summary(model)
	# from plot import plot_running_aegan

	# plot_running_aegan(losses, model_dir)
	return model
