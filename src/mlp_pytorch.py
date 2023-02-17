import torch
from torch import nn, optim
import math

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time

# setup tensorboard stuff
layout = {
    "Multi": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
writer.add_custom_scalars(layout)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_size, out_dim, act_fn):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim
        self.act_fn = act_fn

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            self.act_fn,
            nn.Dropout(p=self.dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hl_size),
            self.act_fn,
            nn.Dropout(p=self.dropout),
        )

        self.fc3 = nn.Sequential(nn.Linear(self.hl_size, self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def earth_mover_distance(y_true, y_pred):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
        dim=-1,
    )


# Select activation function from hyperparams inputs
class ActivationSwitch:
    def fn(self, activation):
        default = nn.PReLU()
        return getattr(
            self, f"activation_function_{activation.lower()}", lambda: default
        )()

    def activation_function_relu(self):
        return nn.ReLU()

    def activation_function_prelu(self):
        return nn.PReLU()

    def activation_function_tanh(self):
        return nn.Tanh()

    def activation_function_sigmoid(self):
        return nn.Sigmoid()

    def activation_function_elu(self):
        return nn.ELU()

    def activation_function_leakyrelu(self):
        return nn.LeakyReLU()

    def activation_function_selu(self):
        return nn.SELU()


def train_mlp(x, y, hyperparams, n_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    activation_switch = ActivationSwitch()
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

    mlp = MLP(
        n_in,
        hyperparams["hl_ini_dim"],
        hyperparams["dropout"],
        int(hyperparams["hl_ini_dim"] * hyperparams["hl_shrink"]),
        out_dim,
        act_fn,
    )
    mlp.to(device)
    mlp.apply(weight_init)
    mlp.train()
    optimizer = optim.Adam(mlp.parameters(), lr=hyperparams["lr"])

    criterion = nn.MSELoss()

    total_step = 0
    for epoch in range(n_epoch):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            # print(total_step % n_noise)
            # if total_step % 10 == 0:
            # print('add noise')
            noise = torch.randn_like(inputs) * 0.3
            inputs = noise + inputs
            # else:
            #     print('no noise')

            optimizer.zero_grad()
            logps = mlp(inputs)

            loss = criterion(logps, labels)
            loss.mean().backward()
            optimizer.step()
            total_step += 1

            running_loss += loss.item()

        valid_loss = 0
        mlp.eval()
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            target = mlp(inputs)

            loss = criterion(target, labels)
            valid_loss += loss.item()

        print("Training loss:", running_loss / len(trainloader))
        print("Validation loss:", valid_loss / len(validloader))

        writer.add_scalar("loss/train", (running_loss / len(trainloader)), epoch)
        writer.add_scalar("loss/validation", (valid_loss / len(validloader)), epoch)
    print("total step =", total_step)

    writer.close()
    return mlp, running_loss / len(trainloader)
