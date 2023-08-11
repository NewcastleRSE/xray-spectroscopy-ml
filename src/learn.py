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
import pickle
import tempfile
import time
from glob import glob
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import model_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setup tensorboard stuff
layout = {
    "Multi": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
writer.add_custom_scalars(layout)

total_step = 0


def log_scalar(name, value, epoch):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, epoch)
    mlflow.log_metric(name, value)


def train(
        x,
        y,
        exp_name,
        model_mode,
        hyperparams,
        n_epoch,
        weight_seed,
        scheduler_lr,
        model_eval,
        load_guess,
        loadguess_params,
):
    EXPERIMENT_NAME = f"{exp_name}"
    RUN_NAME = f"run_{datetime.today()}"

    try:
        EXPERIMENT_ID = mlflow.get_experiment_by_name(
            EXPERIMENT_NAME).experiment_id
    except:
        EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

    out_dim = y[0].size
    n_in = x.shape[1]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    activation_switch = model_utils.ActivationSwitch()
    act_fn = activation_switch.fn(hyperparams["activation"])

    if model_eval:
        # Data split: train/valid/test
        train_ratio = 0.75
        test_ratio = 0.15
        eval_ratio = 0.10

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=1 - train_ratio, random_state=42
        )

        X_test, X_eval, y_test, y_eval = train_test_split(
            X_test, y_test, test_size=eval_ratio / (eval_ratio + test_ratio)
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
    )

    validset = torch.utils.data.TensorDataset(X_test, y_test)
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
    )

    if model_eval:
        evalset = torch.utils.data.TensorDataset(X_eval, y_eval)
        evalloader = torch.utils.data.DataLoader(
            evalset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
        )

    if model_mode == "mlp":
        from model import MLP

        if load_guess:
            import freeze_fn

            model_dir = loadguess_params["model_dir"]
            freeze_params = loadguess_params["freeze_mlp_params"]
            model = freeze_fn.freeze_layers(model_dir, model_mode, freeze_params)
        else:
            model = MLP(
                n_in,
                hyperparams["hl_ini_dim"],
                hyperparams["dropout"],
                hyperparams["n_hl"],
                hyperparams["hl_shrink"],
                out_dim,
                act_fn,
            )

    elif model_mode == "cnn":
        from model import CNN

        if load_guess:
            import freeze_fn

            model_dir = loadguess_params["model_dir"]
            freeze_params = loadguess_params["freeze_cnn_params"]
            model = freeze_fn.freeze_layers(model_dir, model_mode, freeze_params)

        else:
            model = CNN(
                n_in,
                hyperparams["out_channel"],
                hyperparams["channel_mul"],
                hyperparams["hidden_layer"],
                out_dim,
                hyperparams["dropout"],
                hyperparams["kernel_size"],
                hyperparams["stride"],
                act_fn,
                hyperparams["n_cl"],
            )

    elif model_mode == "lstm":
        from model import LSTM

        if load_guess:
            model_dir = loadguess_params["model_dir"]
            model = torch.load(model_dir, map_location=torch.device("cpu"))
            i = 0
            for name, param in model.named_parameters():
                i = i + 1
            num_layers = i
            num_freeze = loadguess_params["n_freeze"]
            if num_freeze > 0:
                i = 0
                for name, param in model.named_parameters():
                    if i < (num_layers-(num_freeze*2)):
                        param.requires_grad = False
                    else:
                        continue
                    i = i+1
        else:
            model = LSTM(
                n_in,
                hyperparams["hidden_size"],
                hyperparams["num_layers"],
                hyperparams["dropout"],
                hyperparams["hl_ini_dim"],
                out_dim,
                act_fn,
            )

    model.to(device)

    if load_guess == False:
        # Model weight & bias initialisation
        kernel_init = model_utils.WeightInitSwitch().fn(
            hyperparams["kernel_init"])
        bias_init = model_utils.WeightInitSwitch().fn(hyperparams["bias_init"])

        # set seed
        torch.cuda.manual_seed(
            weight_seed
        ) if torch.cuda.is_available() else torch.manual_seed(weight_seed)
        model.apply(
            lambda m: model_utils.weight_bias_init(
                m=m, kernel_init_fn=kernel_init, bias_init_fn=bias_init
            )
        )

    optim_fn = model_utils.OptimSwitch().fn(hyperparams["optim_fn"])
    optimizer = optim_fn(model.parameters(), lr=hyperparams["lr"])

    if scheduler_lr["scheduler"]:
        scheduler = model_utils.LRScheduler(
            optimizer,
            scheduler_type=scheduler_lr["scheduler_type"],
            params=scheduler_lr["scheduler_param"],
        )

    # Select loss function
    loss_fn = hyperparams["loss"]["loss_fn"]
    loss_args = hyperparams["loss"]["loss_args"]
    criterion = model_utils.LossSwitch().fn(loss_fn, loss_args)

    # Regularisation of loss function
    loss_reg_type = hyperparams["loss"]["loss_reg_type"]
    loss_reg = True if loss_reg_type is not None else False
    lambda_reg = hyperparams["loss"]["loss_reg_param"]
 
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME):
        mlflow.log_params(hyperparams)
        mlflow.log_param("n_epoch", n_epoch)

        # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()

        for epoch in range(n_epoch):
            print(f">>> epoch = {epoch}")
            model.train()
            running_loss = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.float(), labels.float()

                optimizer.zero_grad()
                logps = model(inputs)

                loss = criterion(logps, labels)

                if loss_reg:
                    l_reg = model_utils.loss_reg_fn(model, loss_reg_type, device)
                    loss += lambda_reg * l_reg

                loss.mean().backward()
                optimizer.step()

                running_loss += loss.item()

            valid_loss = 0
            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.float(), labels.float()

                target = model(inputs)

                loss = criterion(target, labels)
                valid_loss += loss.item()

            if scheduler_lr["scheduler"]:
                before_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                after_lr = optimizer.param_groups[0]["lr"]
                print("Epoch %d: Adam lr %.5f -> %.5f" %
                      (epoch, before_lr, after_lr))

            print("Training loss:", running_loss / len(trainloader))
            print("Validation loss:", valid_loss / len(validloader))

            log_scalar("loss/train", (running_loss / len(trainloader)), epoch)
            log_scalar("loss/validation",
                       (valid_loss / len(validloader)), epoch)

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )

        # Log the model as an artifact of the MLflow run.
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(
            model, artifact_path="pytorch-model", pickle_module=pickle
        )
        print(
            "\nThe model is logged at:\n%s"
            % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
        )

        loaded_model = mlflow.pytorch.load_model(
            mlflow.get_artifact_uri("pytorch-model")
        )

        # Perform model evaluation using invariance tests
        if model_eval:
            import core_eval

            eval_results = core_eval.run_model_eval_tests(
                model, model_mode, trainloader, validloader, evalloader, n_in, out_dim
            )

            # Log results
            for k, v in eval_results.items():
                mlflow.log_dict(v, f"{k}.yaml")

    writer.close()
    return model, running_loss / len(trainloader)
