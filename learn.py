import torch
from torch import nn, optim
import math

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import model_utils

# setup tensorboard stuff
layout = {
    "Multi": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
writer.add_custom_scalars(layout)


def train(x, y, model_mode, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    activation_switch = model_utils.ActivationSwitch()
    act_fn = activation_switch.fn(hyperparams["activation"])

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=hyperparams["batch_size"]
    )

    validset = torch.utils.data.TensorDataset(X_test, y_test)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=hyperparams["batch_size"]
    )

    if model_mode == "mlp":
        from model import MLP

        model = MLP(
            n_in,
            hyperparams["hl_ini_dim"],
            hyperparams["dropout"],
            int(hyperparams["hl_ini_dim"] * hyperparams["hl_shrink"]),
            out_dim,
            act_fn,
        )

    elif model_mode == "cnn":
        from model import CNN

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
        )

    model.to(device)
    model.apply(model_utils.weight_init)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    criterion = nn.MSELoss()

    total_step = 0
    for epoch in range(n_epoch):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            # print(total_step % n_noise)
            if total_step % 20 == 0:
                noise = torch.randn_like(inputs) * 0.3
                inputs = noise + inputs

            optimizer.zero_grad()
            logps = model(inputs)

            loss = criterion(logps, labels)
            loss.mean().backward()
            optimizer.step()
            total_step += 1

            running_loss += loss.item()

        valid_loss = 0
        model.eval()
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            target = model(inputs)

            loss = criterion(target, labels)
            valid_loss += loss.item()

        print("Training loss:", running_loss / len(trainloader))
        print("Validation loss:", valid_loss / len(validloader))

        writer.add_scalar("loss/train", (running_loss / len(trainloader)), epoch)
        writer.add_scalar("loss/validation", (valid_loss / len(validloader)), epoch)
    print("total step =", total_step)

    writer.close()
    return model, running_loss / len(trainloader)
