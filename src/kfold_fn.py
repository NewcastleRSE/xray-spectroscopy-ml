import model_utils
from sklearn.model_selection import RepeatedKFold
import time
import torch
import numpy as np
from learn import train
from ae_learn import train as ae_train
from aegan_learn import train_aegan as aegan_train


def kfold_init(kfold_params, rng):
	kfold_spooler = RepeatedKFold(
		n_splits=kfold_params["n_splits"],
		n_repeats=kfold_params["n_repeats"],
		random_state=rng,
	)
	fit_time = []
	prev_score = 1e6
	loss_fn = kfold_params["loss"]["loss_fn"]
	loss_args = kfold_params["loss"]["loss_args"]
	kfold_loss_fn = model_utils.LossSwitch().fn(loss_fn, loss_args)

	return kfold_spooler, fit_time, kfold_loss_fn, prev_score


def kfold_train(
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
):
	kfold_spooler, fit_time, kfold_loss_fn, prev_score = kfold_init(
		kfold_params, rng)
	# K-fold Cross Validation model evaluation
	train_score = []
	test_score = []
	for fold, (train_index, test_index) in enumerate(kfold_spooler.split(x)):
		print(">> fitting neural net...")
		# Training
		start = time.time()
		model, score = train(
			x[train_index],
			y[train_index],
			exp_name,
			model_mode,
			hyperparams,
			epochs,
			weight_seed,
			lr_scheduler,
			model_eval,
		)
		train_score.append(score)
		fit_time.append(time.time() - start)
		# Testing
		model.eval()
		x_test = torch.from_numpy(x[test_index]).float()
		pred_xanes = model(x_test)
		pred_score = kfold_loss_fn(torch.tensor(
			y[test_index]), pred_xanes).item()
		test_score.append(pred_score)
		if pred_score < prev_score:
			best_model = model
		prev_score = pred_score
	result = {
		"fit_time": fit_time,
		"train_score": train_score,
		"test_score": test_score,
	}
	return result, best_model


def kfold_ae_train(
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
):
	kfold_spooler, fit_time, kfold_loss_fn, prev_score = kfold_init(
		kfold_params, rng)
	train_score = []
	test_recon_score = []
	test_pred_score = []
	for fold, (train_index, test_index) in enumerate(kfold_spooler.split(x)):
		print(">> fitting neural net...")
		# Training
		start = time.time()
		model, score = ae_train(
			x[train_index],
			y[train_index],
			exp_name,
			model_mode,
			hyperparams,
			epochs,
			weight_seed,
			lr_scheduler,
			model_eval,
		)
		train_score.append(score)
		fit_time.append(time.time() - start)
		# Testing
		model.eval()
		x_test = torch.from_numpy(x[test_index]).float()
		y_test = torch.from_numpy(y[test_index]).float()
		recon_x, pred_y = model(x_test)
		recon_score = kfold_loss_fn(x_test, recon_x).item()
		pred_score = kfold_loss_fn(y_test, pred_y).item()
		test_recon_score.append(recon_score)
		test_pred_score.append(pred_score)
		mean_score = np.mean([recon_score, pred_score])
		if mean_score < prev_score:
			best_model = model
		prev_score = mean_score
	result = {
		"fit_time": fit_time,
		"train_score": train_score,
		"test_recon_score": test_recon_score,
		"test_pred_score": test_pred_score,
	}
	return result, best_model


def kfold_aegan_train(
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
):
	kfold_spooler, fit_time, kfold_loss_fn, prev_score = kfold_init(
		kfold_params, rng)
	# K-fold Cross Validation model evaluation
	train_score = []
	test_recon_xyz_score = []
	test_recon_xanes_score = []
	test_pred_xyz_score = []
	test_pred_xanes_score = []
	for fold, (train_index, test_index) in enumerate(kfold_spooler.split(xyz)):
		print(">> fitting neural net...")
		# Training
		start = time.time()
		model, score = aegan_train(
			xyz[train_index],
			xanes[train_index],
			exp_name,
			hyperparams,
			epochs,
			lr_scheduler,
			weight_seed,
			model_eval,
		)
		train_score.append(score["train_loss"][-1])
		fit_time.append(time.time() - start)
		# Testing
		model.eval()
		xyz_test = torch.from_numpy(xyz[test_index]).float()
		xanes_test = torch.from_numpy(xanes[test_index]).float()
		(
			recon_xyz,
			recon_xanes,
			pred_xyz,
			pred_xanes,
		) = model.reconstruct_all_predict_all(xyz_test, xanes_test)
		recon_xyz_score = kfold_loss_fn(xyz_test, recon_xyz).item()
		recon_xanes_score = kfold_loss_fn(xanes_test, recon_xanes).item()
		pred_xyz_score = kfold_loss_fn(xyz_test, pred_xyz).item()
		pred_xanes_score = kfold_loss_fn(xanes_test, pred_xanes).item()
		test_recon_xyz_score.append(recon_xyz_score)
		test_recon_xanes_score.append(recon_xanes_score)
		test_pred_xyz_score.append(pred_xyz_score)
		test_pred_xanes_score.append(pred_xanes_score)
		mean_score = np.mean(
			[
				recon_xyz_score,
				recon_xanes_score,
				pred_xyz_score,
				pred_xanes_score,
			]
		)
		if mean_score < prev_score:
			best_model = model
		prev_score = mean_score

	result = {
		"fit_time": fit_time,
		"train_score": train_score,
		"test_recon_xyz_score": test_recon_xyz_score,
		"test_recon_xanes_score": test_recon_xanes_score,
		"test_pred_xyz_score": test_pred_xyz_score,
		"test_pred_xanes_score": test_pred_xanes_score,
	}
	return result, best_model
