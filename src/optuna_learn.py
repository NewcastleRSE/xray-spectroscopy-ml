import optuna

from learn import train
from ae_learn import train as ae_train
from aegan_learn import train_aegan as aegan_train

def optuna_defaults():
    
    options = {}

    # Generic options (see model_utils.py for available options)
    options["optim_fn"] = ["Adam", "SGD", "RMSprop"]
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
    options["n_cl_min"] = 1
    options["n_cl_max"] = 5
    options["out_channel"] = [8, 16, 32, 64]
    options["channel_mul"] = [2, 3, 4]

    return options
  

def main(optuna_params, x, y, exp_name, model_mode, hyperparams, epochs, weight_seed, lr_scheduler, model_eval):

    n_trials = optuna_params["n_trials"]

    func = lambda trial: optuna_train(trial, x, y, exp_name, model_mode, hyperparams, epochs, weight_seed, lr_scheduler, model_eval,optuna_params)

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
    optuna_params,
    ):

    options = optuna_defaults()

    # Suggest hyperparameters for the trial
    if optuna_params["tune_optim_fn"]:
        hyperparams["optim_fn"] = trial.suggest_categorical("optim_fn", options["optim_fn"])

    if optuna_params["tune_batch_size"]:
        hyperparams["batch_size"] = trial.suggest_categorical("batch_size", options["batch_size"])

    if optuna_params["tune_activation"]:
        hyperparams["activation"] = trial.suggest_categorical("activation", options["activation"])

    if optuna_params["tune_dropout"]:
        hyperparams["dropout"] = trial.suggest_uniform("dropout", options["dropout_min"], options["dropout_max"])

    # Specific model_mode hyperparams
    if model_mode == "aegan_mlp":

        if optuna_params["tune_hidden_layers"]:
            hyperparams["n_hl_gen"] = trial.suggest_int("n_hl_gen", options["n_hl_min"], options["n_hl_max"])
            hyperparams["n_hl_shared"] = trial.suggest_int("n_hl_shared", options["n_hl_min"], options["n_hl_max"])
            hyperparams["n_hl_dis"] = trial.suggest_int("n_hl_dis", options["n_hl_min"], options["n_hl_max"])

        # if optuna_params["tune_loss_fn"]:
        # 	hyperparams["loss_gen"]["loss_fn"] = trial.suggest_categorical("loss_gen", options("loss_fn"))

        # 	if hyperparams["loss"]["loss_fn"] == "wcc":
        # 		hyperparams["loss"]["loss_args"] = trial.suggest_discrete_uniform("loss_args", options["wccloss_min"], options["wccloss_max"], q = 1)

        if optuna_params["tune_lr"]:
            hyperparams["lr_gen"] = trial.suggest_float("lr", options["lr_min"], options["lr_max"], log=True)
            hyperparams["lr_dis"] = trial.suggest_float("lr", options["lr_min"], options["lr_max"], log=True)

    else:

        if optuna_params["tune_loss_fn"]:
            hyperparams["loss"]["loss_fn"] = trial.suggest_categorical("loss_fn", options["loss_fn"])

            if hyperparams["loss"]["loss_fn"] == "wcc":
                hyperparams["loss"]["loss_args"] = trial.suggest_discrete_uniform("loss_args", options["wccloss_min"], options["wccloss_max"], q = 1)

        if optuna_params["tune_lr"]:
            hyperparams["lr"] = trial.suggest_float("lr", options["lr_min"], options["lr_max"], log=True)

        if model_mode == "mlp" or model_mode == "ae_mlp":
            # MLP params
            if optuna_params["tune_hidden_layers"]:
                hyperparams["n_hl"] = trial.suggest_int("n_hl", options["n_hl_min"], options["n_hl_max"])
                hyperparams["hl_ini_dim"] = trial.suggest_categorical("hl_ini_dim", options["hl_size"])
                hyperparams["hl_shrink"] = trial.suggest_uniform("hl_shrink", options["hl_shrink_min"], options["hl_shrink_max"])

                last_hidden_layer_size = int(hyperparams["hl_ini_dim"] * hyperparams["hl_shrink"] ** (hyperparams["n_hl"] - 1))

                if last_hidden_layer_size < 1:
                    raise optuna.exceptions.TrialPruned("The size of the last hidden layer was less than 1. Terminated Optuna trial for this parameterisation.")

        elif model_mode == "cnn" or model_mode == "ae_cnn":
            # CNN specific hyperparams
            if optuna_params["tune_hidden_layers"]:
                hyperparams["n_cl"] = trial.suggest_int("n_cl", options["n_cl_min"], options["n_cl_max"])
                hyperparams["hidden_layer"] = trial.suggest_categorical("hidden_layer", options["hl_size"])

        elif model_mode == "lstm":
            if optuna_params["tune_hidden_layers"]:
                hyperparams["num_layers"] = trial.suggest_int("n_hl", options["n_hl_min"], options["n_hl_max"])
                hyperparams["hl_ini_dim"] = trial.suggest_categorical("hl_ini_dim", options["hl_size"])

    # Set load_guess to False for tuning
    load_guess = False
    load_guess_params = {}

    if model_mode == "mlp" or model_mode == "cnn" or model_mode == "lstm":

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
            load_guess,
            load_guess_params,
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
            load_guess,
            load_guess_params,
        )
    
    elif model_mode == 'aegan_mlp':

        model, score = aegan_train(
            x,
            y,
            exp_name,
            hyperparams,
            epochs,
            weight_seed,
            lr_scheduler,
            model_eval,
            load_guess,
            load_guess_params,
        )

        score = sum([v[-1] for k,v in score.items()])


    return score
    