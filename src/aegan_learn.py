import os
import pickle
import tempfile
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

from model import AEGANTrainer
import model_utils

# setup tensorboard stuff
layout = {
    "Multi": {
        "total_loss": ["multiline", ["total_loss"]],
        "recon_loss": ["Multiline", ["loss/x", "loss/y"]],
        "pred_loss": ["Multiline", ["loss_p/x", "loss_p/y"]],
    },
}
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
writer.add_custom_scalars(layout)


def log_scalar(name, value, epoch):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, epoch)
    mlflow.log_metric(name, value)


def train_aegan(x, y, exp_name, hyperparams, n_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)

    EXPERIMENT_NAME = f"{exp_name}"
    RUN_NAME = f"run_{datetime.today()}"

    try:
        EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        print(EXPERIMENT_ID)
    except:
        EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
        print(EXPERIMENT_ID)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    n_x_features = x.shape[1]
    n_y_features = y.shape[1]

    dataset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=hyperparams["batch_size"]
    )

    hyperparams["input_size_a"] = n_x_features
    hyperparams["input_size_b"] = n_y_features

    model = AEGANTrainer(
        dim_a=hyperparams["input_size_a"],
        dim_b=hyperparams["input_size_b"],
        hidden_size=hyperparams["hidden_size"],
        dropout=hyperparams["dropout"],
        n_hl_gen=hyperparams["n_hl_gen"],
        n_hl_shared=hyperparams["n_hl_shared"],
        n_hl_dis=hyperparams["n_hl_dis"],
        activation=hyperparams["activation"],
        loss_gen=hyperparams["loss_gen"],
        loss_dis=hyperparams["loss_dis"],
        lr_gen=hyperparams["lr_gen"],
        lr_dis=hyperparams["lr_dis"],
    )

    model.to(device)

    # Model weight & bias initialisation
    weight_seed = hyperparams["weight_init_seed"]
    kernel_init = model_utils.WeightInitSwitch().fn(hyperparams["kernel_init"])
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

    model.train()

    # Select running loss function as generative loss function
    loss_fn = hyperparams["loss_gen"]["loss_fn"]
    loss_args = hyperparams["loss_gen"]["loss_args"]
    criterion = model_utils.LossSwitch().fn(loss_fn, loss_args)

    train_total_loss = [None] * n_epoch
    train_loss_x_recon = [None] * n_epoch
    train_loss_y_recon = [None] * n_epoch
    train_loss_x_pred = [None] * n_epoch
    train_loss_y_pred = [None] * n_epoch

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME):
        mlflow.log_params(hyperparams)
        mlflow.log_param("n_epoch", n_epoch)

        # # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()

        for epoch in range(n_epoch):
            running_loss_recon_x = 0
            running_loss_recon_y = 0
            running_loss_pred_x = 0
            running_loss_pred_y = 0
            running_gen_loss = 0
            running_dis_loss = 0

            for inputs_x, inputs_y in trainloader:
                inputs_x, inputs_y = inputs_x.to(device), inputs_y.to(device)
                inputs_x, inputs_y = inputs_x.float(), inputs_y.float()

                model.gen_update(inputs_x, inputs_y)
                model.dis_update(inputs_x, inputs_y)

                recon_x, recon_y, pred_x, pred_y = model.reconstruct_all_predict_all(
                    inputs_x, inputs_y
                )

                # Track running losses
                running_loss_recon_x += criterion(recon_x, inputs_x)
                running_loss_recon_y += criterion(recon_y, inputs_y)
                running_loss_pred_x += criterion(pred_x, inputs_x)
                running_loss_pred_y += criterion(pred_y, inputs_y)

                loss_gen_total = (
                    running_loss_recon_x
                    + running_loss_recon_y
                    + running_loss_pred_x
                    + running_loss_pred_y
                )
                loss_dis = model.loss_dis_total

                running_gen_loss += loss_gen_total.item()
                running_dis_loss += loss_dis.item()

            running_gen_loss = running_gen_loss / len(trainloader)
            running_dis_loss = running_dis_loss / len(trainloader)

            running_loss_recon_x = running_loss_recon_x.item() / len(trainloader)
            running_loss_recon_y = running_loss_recon_y.item() / len(trainloader)
            running_loss_pred_x = running_loss_pred_x.item() / len(trainloader)
            running_loss_pred_y = running_loss_pred_y.item() / len(trainloader)

            log_scalar("gen_loss", running_gen_loss, epoch)
            log_scalar("dis_loss", running_dis_loss, epoch)
            log_scalar("recon_x_loss", running_loss_recon_x, epoch)
            log_scalar("recon_y_loss", running_loss_recon_y, epoch)
            log_scalar("pred_x_loss", running_loss_pred_x, epoch)
            log_scalar("pred_y_loss", running_loss_pred_y, epoch)

            train_loss_x_recon[epoch] = running_loss_recon_x
            train_loss_y_recon[epoch] = running_loss_recon_y
            train_loss_x_pred[epoch] = running_loss_pred_x
            train_loss_y_pred[epoch] = running_loss_pred_y

            train_total_loss[epoch] = running_gen_loss

            print(f">>> Epoch {epoch}...")
            print(
                f">>> Running reconstruction loss (structure) = {running_loss_recon_x:.4f}"
            )
            print(
                f">>> Running reconstruction loss (spectrum) =  {running_loss_recon_y:.4f}"
            )
            print(
                f">>> Running prediction loss (structure) =     {running_loss_pred_x:.4f}"
            )
            print(
                f">>> Running prediction loss (spectrum) =      {running_loss_pred_y:.4f}"
            )

            losses = {
                "train_loss": train_total_loss,
                "loss_x_recon": train_loss_x_recon,
                "loss_y_recon": train_loss_y_recon,
                "loss_x_pred": train_loss_x_pred,
                "loss_y_pred": train_loss_y_pred,
            }

    return model, losses
