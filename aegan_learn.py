import torch
from torch import nn

from model import AEGANTrainer
from model_utils import weight_init


def train_aegan(x, y, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    n_x_features = x.shape[1]
    n_y_features = y.shape[1]

    dataset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64)

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

    # Initialise weights
    model.apply(weight_init)

    model.train()
    loss_fn = nn.MSELoss()

    train_total_loss = [None] * n_epoch
    train_loss_x_recon = [None] * n_epoch
    train_loss_y_recon = [None] * n_epoch
    train_loss_x_pred = [None] * n_epoch
    train_loss_y_pred = [None] * n_epoch

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
            running_loss_recon_x += loss_fn(recon_x, inputs_x)
            running_loss_recon_y += loss_fn(recon_y, inputs_y)
            running_loss_pred_x += loss_fn(pred_x, inputs_x)
            running_loss_pred_y += loss_fn(pred_y, inputs_y)

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

    return losses, model
