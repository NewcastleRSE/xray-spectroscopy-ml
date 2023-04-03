import optuna

from learn import train
from ae_learn import train as ae_train
from aegan_learn import train_aegan as aegan_train

def optuna_defaults():

	options = {}

	# Generic options (see model_utils.py for available options)
	options["batch_size"] = [8, 16, 32, 64]
	options["activation"] = ["relu", "prelu", "tanh", "sigmoid", "elu", "leakyrelu", "selu"]

	options["loss_fn"] = ["mse", "emd", "cosine", "l1", "wcc"]
	options["wccloss_min"] = 5
	options["wccloss_max"] = 15 

	options["lr_min"] = 1e-7
	options["lr_max"] = 1e-3
	options["dropout_min"] = 0.2 
	options["dropout_max"] = 0.5

	# MLP Specific
	options["n_hl_min"] = 2
	options["n_hl_max"] = 5
	options["hl_size"] = [64, 128, 256, 512]
	options["hl_shrink_min"] = 0.2 
	options["hl_shrink_max"] = 0.5

	# CNN Specific
	options["out_channel"] = [8, 16, 32, 64]
	options["channel_mul"] = [2, 3, 4]

	return options
  

def optuna_learn(n_trials, x, y, exp_name, model_mode, hyperparams, epochs, weight_seed, lr_scheduler, model_eval):

	func = lambda trial: optuna_train(trial, x, y, exp_name, model_mode, hyperparams, epochs, weight_seed, lr_scheduler, model_eval)

	study = optuna.create_study(direction="minimize")
	study.optimize(func, n_trials=n_trials, timeout=None)

	# Print optuna study statistics
	print(f"{'='*20} Optuna {'='*20}")
	print("Study statistics: ")
	print(f"  Number of finished trials: {len(study.trials)}")

	print("Best trial:")
	trial = study.best_trial

	print(f"  Value: {trial.value}")

	print("  Params: ")
	for k, v in trial.params.items():
		print(f"    {k}: {v}")

	return trial, trial.value


def optuna_train(trial, 
	x,
	y,
	exp_name,
	model_mode,
	hyperparams,
	epochs,
	weight_seed,
	lr_scheduler,
	model_eval,
	):
	
	options = optuna_defaults()

	# Suggest hyperparameters for the trial
	hyperparams["batch_size"] = trial.suggest_categorical("batch_size", options["batch_size"])
	hyperparams["activation"] = trial.suggest_categorical("activation", options["activation"])

	if model_mode == "mlp" or "ae_mlp":

		# MLP params
		hyperparams["n_hl"] = trial.suggest_int("n_hl", options["n_hl_min"], options["n_hl_max"])
		hyperparams["hl_ini_dim"] = trial.suggest_categorical("hl_ini_dim", options["hl_size"])
		hyperparams["hl_shrink"] = trial.suggest_uniform("hl_shrink", options["hl_shrink_min"], options["hl_shrink_max"])

	elif model_mode == "cnn" or model_mode == "ae_cnn":

		# CNN specific hyperparams
		hyperparams["hidden_layer"] = trial.suggest_categorical("hidden_layer", options["hl_size"])


	# Generic hyperparams
	hyperparams["loss"]["loss_fn"] = trial.suggest_categorical("loss_fn", options["loss_fn"])

	if hyperparams["loss"]["loss_fn"] == "wcc":
		hyperparams["loss"]["loss_args"] = trial.suggest_discrete_uniform("loss_args", options["wccloss_min"], options["wccloss_max"], q = 1)


	hyperparams["lr"] = trial.suggest_float("lr", options["lr_min"], options["lr_max"], log=True)
	hyperparams["dropout"] = trial.suggest_uniform("dropout", options["dropout_min"], options["dropout_max"])



	# elif model_mode == "aegan_mlp":

	# 	hyperparams["n_hl_gen"] = trial.suggest_categorical("n_hl_gen", options("n_hl"))
	# 	hyperparams["n_hl_shared"] = trial.suggest_categorical("n_hl_shared", options("n_hl"))
	# 	hyperparams["n_hl_dis"] = trial.suggest_categorical("n_hl_dis", options("n_hl"))

	# 	hyperparams["loss_gen"]["loss_fn"] = trial.suggest_categorical("loss_gen", options("loss_fn"))

	# 	hyperparams["lr_gen"] = trial.suggest_float("lr", options["lr_min"], options["lr_max"], log=True)
	# 	hyperparams["lr_dis"] = trial.suggest_float("lr", options["lr_min"], options["lr_max"], log=True)

		
		

	if model_mode == "mlp" or model_mode == "cnn":

		model, score = train(
			x,
			y,
			exp_name,
			model_mode,
			hyperparams,
			epochs,
			weight_seed,
			lr_scheduler,
			model_eval,
			)
	elif model_mode == 'ae_mlp' or model_mode == 'ae_cnn':

		model, score = ae_train(
			x,
			y,
			exp_name,
			model_mode,
			hyperparams,
			epochs,
			weight_seed,
			lr_scheduler,
			model_eval,
		)
	
	# elif model_mode == 'aegan_mlp':


	return score


# import os
# import tempfile
# import time
# from datetime import datetime
# import pickle
# from sklearn.model_selection import train_test_split

# import torch
# from torch import nn, optim
# from torch.utils.tensorboard import SummaryWriter
# import mlflow
# import mlflow.pytorch
# import optuna

# import model_utils

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # setup tensorboard stuff
# layout = {
#     "Multi": {
#         "loss": ["Multiline", ["loss/train", "loss/validation"]],
#     },
# }
# writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
# writer.add_custom_scalars(layout)

# total_step = 0


# def log_scalar(name, value, epoch):
#     """Log a scalar value to both MLflow and TensorBoard"""
#     writer.add_scalar(name, value, epoch)
#     mlflow.log_metric(name, value)


# def train(trial, x, y, exp_name, model_mode, hyperparams, n_epoch):
#     EXPERIMENT_NAME = f"{exp_name}"
#     RUN_NAME = f"run_{datetime.today()}"

#     try:
#         EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
#         print(EXPERIMENT_ID)
#     except:
#         EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
#         print(EXPERIMENT_ID)

#     out_dim = y[0].size
#     n_in = x.shape[1]

#     x = torch.from_numpy(x)
#     y = torch.from_numpy(y)

#     activation_switch = model_utils.ActivationSwitch()
#     act_fn = activation_switch.fn(hyperparams["activation"])

#     X_train, X_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.2, random_state=42
#     )

#     trainset = torch.utils.data.TensorDataset(X_train, y_train)
#     trainloader = torch.utils.data.DataLoader(
#         trainset,
#         batch_size=hyperparams["batch_size"],
#         shuffle=True,
#     )

#     validset = torch.utils.data.TensorDataset(X_test, y_test)
#     validloader = torch.utils.data.DataLoader(
#         validset,
#         batch_size=hyperparams["batch_size"],
#         shuffle=False,
#     )

#     if model_mode == "mlp":
#         from model import MLP

#         model = MLP(
#             n_in,
#             hyperparams["hl_ini_dim"],
#             hyperparams["dropout"],
#             int(hyperparams["hl_ini_dim"] * hyperparams["hl_shrink"]),
#             out_dim,
#             act_fn,
#         )

#     elif model_mode == "cnn":
#         from model import CNN

#         model = CNN(
#             n_in,
#             hyperparams["out_channel"],
#             hyperparams["channel_mul"],
#             hyperparams["hidden_layer"],
#             out_dim,
#             hyperparams["dropout"],
#             hyperparams["kernel_size"],
#             hyperparams["stride"],
#             act_fn,
#         )

#     model.to(device)

#     # Model weight & bias initialisation
#     weight_seed = hyperparams["weight_init_seed"]
#     kernel_init = model_utils.WeightInitSwitch().fn(hyperparams["kernel_init"])
#     bias_init = model_utils.WeightInitSwitch().fn(hyperparams["bias_init"])

#     # set seed
#     torch.cuda.manual_seed(
#         weight_seed
#     ) if torch.cuda.is_available() else torch.manual_seed(weight_seed)
#     model.apply(
#         lambda m: model_utils.weight_bias_init(
#             m=m, kernel_init_fn=kernel_init, bias_init_fn=bias_init
#         )
#     )

#     optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
#     lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
#     optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

#     # n_epoch = trial.suggest_int("epoch", 1, 5)

#     model.train()
#     # optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

#     # Select loss function
#     loss_fn = hyperparams["loss"]["loss_fn"]
#     loss_args = hyperparams["loss"]["loss_args"]
#     criterion = model_utils.LossSwitch().fn(loss_fn, loss_args)

#     with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME):
#         mlflow.log_params(hyperparams)
#         mlflow.log_param("n_epoch", n_epoch)

#         # # Create a SummaryWriter to write TensorBoard events locally
#         output_dir = dirpath = tempfile.mkdtemp()

#         for epoch in range(n_epoch):
#             running_loss = 0
#             for inputs, labels in trainloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 inputs, labels = inputs.float(), labels.float()

#                 # print(total_step % n_noise)
#                 # if total_step % 20 == 0:
#                 #     noise = torch.randn_like(inputs) * 0.3
#                 #     inputs = noise + inputs

#                 optimizer.zero_grad()
#                 logps = model(inputs)

#                 loss = criterion(logps, labels)
#                 loss.mean().backward()
#                 optimizer.step()
#                 # total_step += 1

#                 running_loss += loss.item()

#             valid_loss = 0
#             model.eval()
#             for inputs, labels in validloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 inputs, labels = inputs.float(), labels.float()

#                 target = model(inputs)

#                 loss = criterion(target, labels)
#                 valid_loss += loss.item()

#             print("Training loss:", running_loss / len(trainloader))
#             print("Validation loss:", valid_loss / len(validloader))

#             log_scalar("loss/train", (running_loss / len(trainloader)), epoch)
#             log_scalar("loss/validation", (valid_loss / len(validloader)), epoch)
#         # print("total step =", total_step)

#         # Upload the TensorBoard event logs as a run artifact
#         print("Uploading TensorBoard events as a run artifact...")
#         mlflow.log_artifacts(output_dir, artifact_path="events")
#         print(
#             "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
#             % os.path.join(mlflow.get_artifact_uri(), "events")
#         )

#         # Log the model as an artifact of the MLflow run.
#         print("\nLogging the trained model as a run artifact...")
#         mlflow.pytorch.log_model(
#             model, artifact_path="pytorch-model", pickle_module=pickle
#         )
#         print(
#             "\nThe model is logged at:\n%s"
#             % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
#         )

#         loaded_model = mlflow.pytorch.load_model(
#             mlflow.get_artifact_uri("pytorch-model")
#         )

#     writer.close()
#     loss = running_loss / len(trainloader)
#     trial.report(loss, epoch)
#     return loss


# def optuna_learn(n_trials, x, y, exp_name, model_mode, hyperparams, n_epoch):
#     func = lambda trial: train(trial, x, y, exp_name, model_mode, hyperparams, n_epoch)

#     study = optuna.create_study(direction="minimize")
#     study.optimize(func, n_trials=n_trials, timeout=None)

#     # Print optuna study statistics
#     print(f"{'='*20} Optuna {'='*20}")
#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))

#     return trial, trial.value