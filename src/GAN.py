from pathlib import Path
from utils import list_filestems
from utils import linecount
from structure.rdc import RDC
from structure.wacsf import WACSF
from inout import load_xyz
from inout import load_xanes
from utils import unique_path
from inout import save_xanes
from spectrum.xanes import XANES

import os
import torchinfo
import torch
import torch.nn as nn
import tqdm as tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epoch = 500
learning_rate = 0.0001
wdecay = 0.001


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size=97, hidden_size=100, num_layers=2, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * 100, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 226),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        out = self.fc(x)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.lstm = nn.LSTM(input_size=226, hidden_size=100, num_layers=2, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * 100, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        realfake = self.fc(x)

        return realfake 


x_path = "/home/tom/XAS-3dtm/original/fe/preconv/xyz_train"
y_path = "/home/tom/XAS-3dtm/original/fe/preconv/xanes_train"

xyz_path = Path(x_path)
xanes_path = Path(y_path)

ids = list(
    set(list_filestems(xyz_path))
    & set(list_filestems(xanes_path))

)

ids.sort()

descriptors = {"rdc": RDC, "wacsf": WACSF}
desc_params = {'r_min': 1.0, 'r_max': 6.0, 'n_g2': 32, 'n_g4': 64, 'z':[1,2,4,8,16,32,64,128]}

descriptor = descriptors.get("wacsf")(
    **desc_params
)

n_samples = len(ids)
n_x_features = descriptor.get_len()
n_y_features = linecount(
    xanes_path / f"{ids[0]}.txt") - 2

xyz_data = np.full((n_samples, n_x_features), np.nan)
print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
xanes_data = np.full((n_samples, n_y_features), np.nan)
print(">> preallocated {}x{} array for Y data...".format(
    *xanes_data.shape))
print(">> ...everything preallocated!\n")


print(">> loading data into array(s)...")
for i, id_ in enumerate(tqdm.tqdm(ids)):
    with open(xyz_path / f"{id_}.xyz", "r") as f:
        atoms = load_xyz(f)
    xyz_data[i, :] = descriptor.transform(atoms)
    with open(xanes_path / f"{id_}.txt", "r") as f:
        xanes = load_xanes(f)
    e, xanes_data[i, :] = xanes.spectrum
print(">> ...loaded into array(s)!\n")

out_dim = xanes_data[0].size
n_in = xyz_data.shape[1]

x = torch.from_numpy(xyz_data)
y = torch.from_numpy(xanes_data)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

trainset = torch.utils.data.TensorDataset(x, y)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
)

validset = torch.utils.data.TensorDataset(X_test, y_test)
validloader = torch.utils.data.DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
)

gen = Generator().to(device)
dis = Discriminator().to(device)

optimizer_Gen = torch.optim.Adam(
    gen.parameters(),
    lr=learning_rate,
    weight_decay=wdecay
)
optimizer_Dis = torch.optim.Adam(
    dis.parameters(),
    lr=learning_rate,
    weight_decay=wdecay
)
adversarial_loss = nn.BCEWithLogitsLoss()

real_label = 1
fake_label = 0

# Print the model summary
torchinfo.summary(gen)
torchinfo.summary(dis)

for epoch in range(n_epoch):
    print(f">>> epoch = {epoch}")
    gen.train()
    dis.train()
    running_loss = 0

    for xyz, xanes in trainloader:
        xyz, xanes = xyz.to(device), xanes.to(device)
        xyz, xanes = xyz.float(), xanes.float()

        label = torch.ones(xyz.shape[0], 1).to(device)

        # train discriminator
        # train with real
        optimizer_Dis.zero_grad()
        output_real = dis(xanes)

        # Calculate error and backpropagate
        errD_real = adversarial_loss(output_real, label)
        errD_real.backward()

        # train with fake
        # Generate fake data
        fake_xanes = gen(xyz)
        label = torch.zeros(xyz.shape[0], 1).to(device)

        output_fake = dis(fake_xanes)
        errD_fake = adversarial_loss(output_fake, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizer_Dis.step()

        # train generator
        fake_xanes = gen(xyz)
        label = torch.ones(xyz.shape[0], 1).to(device)
        # Reset gradients
        optimizer_Gen.zero_grad()

        output = dis(fake_xanes)
        errG = adversarial_loss(output, label)
        errG.backward()
        optimizer_Gen.step()

    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " %
          (epoch, n_epoch, i, len(trainloader), errD.item(), errG.item(), ))

# generate the data
gen.eval()

x_path = "/home/tom/XAS-3dtm/original/fe/preconv/xyz_test"
y_path = "/home/tom/XAS-3dtm/original/fe/preconv/xanes_test"
xyz_path = Path(x_path)
xanes_path = Path(y_path)

ids = list(
    set(list_filestems(xyz_path))
    & set(list_filestems(xanes_path))

)

ids.sort()

descriptors = {"rdc": RDC, "wacsf": WACSF}
desc_params = {'r_min': 1.0, 'r_max': 6.0, 'n_g2': 32, 'n_g4': 64, 'z':[1,2,4,8,16,32,64,128]}

descriptor = descriptors.get("wacsf")(
    **desc_params
)

n_samples = len(ids)
n_x_features = descriptor.get_len()
n_y_features = linecount(
    xanes_path / f"{ids[0]}.txt") - 2

xyz_data = np.full((n_samples, n_x_features), np.nan)
print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
xanes_data = np.full((n_samples, n_y_features), np.nan)
print(">> preallocated {}x{} array for Y data...".format(
    *xanes_data.shape))
print(">> ...everything preallocated!\n")


print(">> loading data into array(s)...")
for i, id_ in enumerate(tqdm.tqdm(ids)):
    with open(xyz_path / f"{id_}.xyz", "r") as f:
        atoms = load_xyz(f)
    xyz_data[i, :] = descriptor.transform(atoms)
    with open(xanes_path / f"{id_}.txt", "r") as f:
        xanes = load_xanes(f)
    e, xanes_data[i, :] = xanes.spectrum
print(">> ...loaded into array(s)!\n")

xyz = torch.from_numpy(xyz_data)
xyz = xyz.float().to(device)

output_directory = "generative"
os.makedirs(output_directory, exist_ok=True)
with torch.no_grad():
    for epoch in range(1):

        fake_xanes = gen(xyz)
        energy = np.tile(e, (fake_xanes.shape[0], 1))

        for id_, fake_xanes_ in tqdm.tqdm(zip(ids, fake_xanes)):

            output_file_path = os.path.join(output_directory, id_ + ".txt")
            with open(output_file_path, "w") as f:
                save_xanes(
                    f, XANES(e, fake_xanes_.detach().cpu().numpy()))

