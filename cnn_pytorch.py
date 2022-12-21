import torch
from torch import nn, optim
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


class CNN(nn.Module):
    def __init__(
        self,
        input_size,
        out_channel,
        channel_mul,
        hidden_layer,
        out_dim,
        dropout,
        kernel_size,
        stride,
        act_fn,
    ):
        super().__init__()

        self.input_size = input_size
        self.out_channel = out_channel
        self.channel_mul = channel_mul
        self.hidden_layer = hidden_layer
        self.out_dim = out_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_fn = act_fn

        self.conv1_shape = int(((self.input_size - self.kernel_size) / self.stride) + 1)
        self.conv2_shape = int(
            ((self.conv1_shape - self.kernel_size) / self.stride) + 1
        )
        print(self.conv1_shape)
        print(self.conv2_shape)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            nn.BatchNorm1d(num_features=self.out_channel),
            self.act_fn,
            nn.Dropout(p=self.dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channel,
                out_channels=self.out_channel * self.channel_mul,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            nn.BatchNorm1d(num_features=self.out_channel * self.channel_mul),
            self.act_fn,
            nn.Dropout(p=self.dropout),
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(
                self.conv2_shape * (self.out_channel * self.channel_mul), self.out_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = self.dense_layer(x)

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


def train_cnn(x, y, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    activation_switch = ActivationSwitch()
    act_fn = activation_switch.fn(hyperparams["activation"])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=hyperparams["batch_size"]
    )

    validset = torch.utils.data.TensorDataset(X_test, y_test)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=hyperparams["batch_size"]
    )

    cnn = CNN(
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

    cnn.to(device)
    cnn.apply(weight_init)
    cnn.train()
    optimizer = optim.Adam(cnn.parameters(), lr=hyperparams["lr"])

    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()
            logps = cnn(inputs)

            loss = criterion(logps, labels)
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss = 0
        cnn.eval()
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            target = cnn(inputs)

            loss = criterion(target, labels)
            valid_loss += loss.item()

        print("Training loss:", running_loss / len(trainloader))
        print("Validation loss:", valid_loss / len(validloader))

        writer.add_scalar("loss/train", (running_loss / len(trainloader)), epoch)
        writer.add_scalar("loss/validation", (valid_loss / len(validloader)), epoch)

    writer.close()

    return cnn, running_loss / len(trainloader)
