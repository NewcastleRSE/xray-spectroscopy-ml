import os
import pickle
import tempfile
import time
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

import model_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setup tensorboard stuff
layout = {
    "Multi": {
        "recon_loss": ["Multiline", ["recon_loss/train", "recon_loss/validation"]],
        "pred_loss": ["Multiline", ["pred_loss/train", "pred_loss/validation"]],
    },
}
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
writer.add_custom_scalars(layout)


def log_scalar(name, value, epoch):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, epoch)
    mlflow.log_metric(name, value)


def train(x, y, exp_name, model_mode, hyperparams, n_epoch):
    EXPERIMENT_NAME = f"{exp_name}"
    RUN_NAME = f"run_{datetime.today()}"

    try:
        EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        print(EXPERIMENT_ID)
    except:
        EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
        print(EXPERIMENT_ID)

    out_dim = y[0].size
    n_in = x.shape[1]

    le = preprocessing.LabelEncoder()

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    activation_switch = model_utils.ActivationSwitch()
    act_fn = activation_switch.fn(hyperparams["activation"])

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

    if model_mode == "ae_mlp":
        from model import AE_mlp

        model = AE_mlp(
            n_in,
            hyperparams["hl_ini_dim"],
            hyperparams["dropout"],
            int(hyperparams["hl_ini_dim"] * hyperparams["hl_shrink"]),
            out_dim,
            act_fn,
        )

    elif model_mode == "ae_cnn":
        from model import AE_cnn

        model = AE_cnn(
            n_in,
            hyperparams["out_channel"],
            hyperparams["channel_mul"],
            hyperparams["hidden_layer"],
            out_dim,
            hyperparams["dropout"],
            hyperparams["kernel_size"],
            hyperparams["stride"],
            act_fn,
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
    optimizer = optim.Adam(
        model.parameters(), lr=hyperparams["lr"], weight_decay=0.0000
    )

    # Select loss function
    loss_fn = hyperparams["loss"]["loss_fn"]
    loss_args = hyperparams["loss"]["loss_args"]
    criterion = model_utils.LossSwitch().fn(loss_fn, loss_args)

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME):
        mlflow.log_params(hyperparams)
        mlflow.log_param("n_epoch", n_epoch)

        # # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()

        total_step = 0
        for epoch in range(n_epoch):
            running_loss = 0
            loss_r = 0
            loss_p = 0

            total_step_train = 0
            for inputs, labels in trainloader:
                inputs, labels = (
                    inputs.to(device),
                    labels.to(device),
                )
                inputs, labels = (
                    inputs.float(),
                    labels.float(),
                )

                # if total_step % 20 == 0:
                #     noise = torch.randn_like(inputs) * 0.3
                #     inputs = noise + inputs

                optimizer.zero_grad()

                recon_input, outputs = model(inputs)

                loss_recon = criterion(recon_input, inputs)
                loss_pred = criterion(outputs, labels)

                loss = loss_recon + loss_pred
                loss.backward()

                optimizer.step()
                running_loss += loss.mean().item()
                loss_r += loss_recon.item()
                loss_p += loss_pred.item()

                # print("train:", loss_recon.item(), loss_pred.item(), loss.item())

                total_step_train += 1

            valid_loss = 0
            valid_loss_r = 0
            valid_loss_p = 0
            model.eval()
            total_step_valid = 0
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.float(), labels.float()

                recon_input, outputs = model(inputs)

                loss_recon = criterion(recon_input, inputs)
                loss_pred = criterion(outputs, labels)

                loss = loss_recon + loss_pred

                valid_loss = loss.item()
                valid_loss_r += loss_recon.item()
                valid_loss_p += loss_pred.item()

                # print("valid:", loss_recon.item(), loss_pred.item(), loss.item())

                total_step_valid += 1

            # print(total_step_train, total_step_valid)
            # print(len(trainloader),len(validloader) )

            print("Training loss:", running_loss / len(trainloader))
            print("Validation loss:", valid_loss / len(validloader))

            log_scalar("total_loss/train", (running_loss / len(trainloader)), epoch)
            log_scalar("total_loss/validation", (valid_loss / len(validloader)), epoch)

            log_scalar("recon_loss/train", (loss_r / len(trainloader)), epoch)
            log_scalar(
                "recon_loss/validation", (valid_loss_r / len(validloader)), epoch
            )

            log_scalar("pred_loss/train", (loss_p / len(trainloader)), epoch)
            log_scalar("pred_loss/validation", (valid_loss_p / len(validloader)), epoch)

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

    writer.close()

    return model, running_loss / len(trainloader)