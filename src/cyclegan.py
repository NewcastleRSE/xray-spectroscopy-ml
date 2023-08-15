#############################################################################
#                                                                           #
# cycleGAN written by Tom Penfold (Aug 2023)                                #
# The generator and discriminator are based upon MLPs                       #
# This reads in two sets of XANES spectra and converts one to the other     #
# The objective is to read in experimental and theoretical spectra          #
# and convert the theoretical spectra so they are more 'experimental' like. #
#                                                                           #
#                                                                           #
# At the moment all the hyperparameters and data paths are hardcoded below  #
# The code is simply run using python3 cyclegan.py                          #
#                                                                           #
#############################################################################

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
import torchinfo
from torchinfo import summary
import os
import random
import torch
import torch.nn as nn
import tqdm as tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################################################################
# Defining the CycleGAN Network #
################################################################################################

# Define the Generator network: Based upon an MLP, where the input and output sizes are the same as its converting spectrum to spectrum
# All hyperparameters are currently 'hardwired'
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(475, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128,475)
        )

    def forward(self, x):
        return self.model(x)


# Define the Discriminator network: Based upon an MLP, where the input and output sizes are the same as its converting spectrum to spectrum
# All hyperparameters are currently 'hardwired'
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(475, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 56),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(56, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Define the CycleGAN model: This is essentially linking the generator and discriminator
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.G_AB = Generator()
        self.G_BA = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()

    def forward(self, real_A, real_B):
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        return fake_A, fake_B, rec_A, rec_B

# Define custom dataset: All this does is convert the read in data into a torch tensor
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

################################################################################################
#                                                                                              #
# MAIN PART OF THE CODE                                                                        #
#                                                                                              #
################################################################################################

################################################################################################
# DATA GENERATION AND SORTING                                                                  #
################################################################################################

# Define your Data Paths 
x_path = "/home/tom/Fe-feff/xanes/"
y_path = "/home/tom/Fe-expt/xanes/"
xyz_path = Path(x_path)
xanes_path = Path(y_path)

# Data parameters, need to refine for your problem 
n_x_features = 475 
n_y_features = n_x_features 
n_x_samples = 36657 
n_y_samples = 4329 

# Network parameters, need to refine for your problem 
learning_rate = 0.00005
wdecay = 0.001
num_epochs = 2000
batch_size = 64 

xyz_data = np.full((n_x_samples, n_x_features), np.nan)
selected_xyz_data = np.full((n_y_samples, n_x_features), np.nan)
print(">> preallocated {}x{} array for X data...".format(*xyz_data.shape))
xanes_data = np.full((n_y_samples, n_y_features), np.nan)
print(">> preallocated {}x{} array for Y data...".format(
    *xanes_data.shape))
print(">> ...everything preallocated!\n")


print(">> loading data into array(s)...")
file_list = [filename for filename in os.listdir(x_path) if filename.endswith(".txt")]
for i, filename in enumerate(tqdm.tqdm(file_list)):
    file_path = os.path.join(x_path, filename)

    with open(file_path, "r") as f:
        xanes = load_xanes(f)
    e, xyz_data[i, :] = xanes.spectrum

file_list = [filename for filename in os.listdir(y_path) if filename.endswith(".txt")]
for i, filename in enumerate(tqdm.tqdm(file_list)):
    file_path = os.path.join(y_path, filename)
    with open(file_path, "r") as f:
        xanes = load_xanes(f)
    e, xanes_data[i, :] = xanes.spectrum

################################################################################################
# Augmenting the experimental spectra to n_y_samples, as there are usually less than to theory # 
################################################################################################
i = 0
while i < (n_y_samples - len(file_list)):
    rand1 = random.choice(range(len(file_list)))
    rand2 = random.choice(range(len(file_list)))
    rand3 = torch.rand(1)
    xanes_data[i+len(file_list), :] = (rand3 * xanes_data[rand1, :]) + ((1 - rand3) * xanes_data[rand2, :])
    i += 1

print(">> ...loaded into array(s) and augmented the experimental data!\n")

# Network hyperparameters
input_dim = n_x_features
output_dim = n_y_features

# Initialize the CycleGAN, disciminator and generator models
model = CycleGAN()

# Define loss functions
criterion_cycle = nn.MSELoss() 
criterion_identity = nn.MSELoss()
criterion_adv = nn.BCEWithLogitsLoss()

# Set device (GPU or CPU)
model = model.to(device)

criterion_cycle = criterion_cycle.to(device)
criterion_identity = criterion_identity.to(device)
criterion_adv = criterion_adv.to(device)

# Initialize optimizers
optimizer_G = optim.Adam(
    itertools.chain(model.G_AB.parameters(), model.G_BA.parameters()), lr=learning_rate, weight_decay=wdecay)
 
optimizer_D_A = optim.Adam(model.D_A.parameters(), lr=learning_rate, weight_decay=wdecay)
optimizer_D_B = optim.Adam(model.D_B.parameters(), lr=learning_rate, weight_decay=wdecay)

xyz_data = torch.Tensor(xyz_data)
xanes_data = torch.Tensor(xanes_data)

dataset_A = MyDataset(xyz_data)
dataset_B = MyDataset(xanes_data)
dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

# Print the model summary
torchinfo.summary(model)

# Train the cycleGAN model
for epoch in range(num_epochs):
 
    loss_G = 0.0
    loss_D_A = 0.0
    loss_A_D = 0.0

    for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Train the generators
        optimizer_G.zero_grad()

        # Identity loss
        loss_identity_A = criterion_identity(model.G_BA(real_A), real_A)
        loss_identity_B = criterion_identity(model.G_AB(real_B), real_B)
        loss_identity = loss_identity_A + loss_identity_B

        # Adversarial loss
        fake_A, fake_B, rec_A, rec_B = model(real_A, real_B)

        output_fake_A = model.D_B(fake_A)
        output_fake_B = model.D_B(fake_B)
        labelA = torch.ones_like(output_fake_A).to(device)
        labelB = torch.ones_like(output_fake_B).to(device)
        loss_GAN_AB = criterion_adv(output_fake_B, labelB)
        loss_GAN_BA = criterion_adv(output_fake_A, labelA)
        loss_GAN = loss_GAN_AB + loss_GAN_BA

        # Cycle-consistency loss
        loss_cycle_A = criterion_cycle(rec_A, real_A)
        loss_cycle_B = criterion_cycle(rec_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # Total generator loss
        loss_G = loss_identity + loss_GAN + loss_cycle
        loss_G.backward()
        optimizer_G.step()

        # Train the discriminators
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        output_real_A = model.D_A(real_A)
        output_real_B = model.D_B(real_B)

        output_fake_A2 = model.D_A(fake_A.detach())
        output_fake_B2 = model.D_A(fake_B.detach())
        
        label_realA = torch.ones_like(output_real_A).to(device)
        label_realB = torch.ones_like(output_real_B).to(device)
        label_fakeA2 = torch.zeros_like(output_fake_A2).to(device)
        label_fakeB2 = torch.zeros_like(output_fake_B2).to(device)

        loss_D_A = criterion_adv(output_real_A, label_realA) + criterion_adv(output_fake_A2, label_fakeA2)
        loss_D_B = criterion_adv(output_real_B, label_realB) + criterion_adv(output_fake_B2, label_fakeB2)

        loss_D_A.backward()
        loss_D_B.backward()

        optimizer_D_A.step()
        optimizer_D_B.step()

    # Print losses
    print(
        "[Epoch %d] [D_A loss: %f] [D_B loss: %f] [G loss: %f]"
        % (
            epoch,
            loss_D_A.item(),
            loss_D_B.item(),
            loss_G.item(),
        )
    )

################################################################################################
#
# Using the model which has been trained and generate new spectra
#
################################################################################################

file_list = [filename for filename in os.listdir(x_path) if filename.endswith(".txt")]
xyz_data_test = np.full((1, n_x_features), np.nan)

# define an output directory
output_directory = "outputs"
os.makedirs(output_directory, exist_ok=True)

for i, filename in enumerate(tqdm.tqdm(file_list)):
    file_path = os.path.join(x_path, filename)

    with open(file_path, "r") as f:
        xanes = load_xanes(f)
    e, xyz_data_test[0, :] = xanes.spectrum

    input_A = torch.Tensor(xyz_data_test).to(device)
    output_B = model.G_AB(input_A)
    output_B_print = output_B.detach().cpu().numpy()

    output_file_path = os.path.join(output_directory, filename)
    with open(output_file_path, "w") as f:
        f.write('Fun Line 1\n')
        f.write('Fun Line 2\n')
        for i, value in enumerate(e):
            f.write(f'{value} {output_B_print[0, i]}\n')
