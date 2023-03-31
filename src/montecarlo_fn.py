import torch
from sklearn.metrics import mean_squared_error
import tqdm as tqdm

from inout import save_xanes_mean, save_xyz_mean
from spectrum.xanes import XANES


def montecarlo_dropout(
    model, input_data, n_mc, data_compress, predict_dir, mode, plot_save
):
    from plot import plot_mc_predict

    model.train()

    prob_output = []

    input_data = torch.from_numpy(input_data)
    input_data = input_data.float()

    print("Running Monte-carlo dropout")
    for t in range(n_mc):
        output_data = model(input_data)
        prob_output.append(output_data)

    prob_mean = torch.mean(torch.stack(prob_output), dim=0)
    prob_var = torch.std(torch.stack(prob_output), dim=0)

    print(
        "MSE y to y prob : ",
        mean_squared_error(data_compress["y"], prob_mean.detach().numpy()),
    )

    if plot_save:
        plot_mc_predict(
            data_compress["ids"],
            data_compress["y"],
            data_compress["y_predict"],
            prob_mean.detach().numpy(),
            prob_var.detach().numpy(),
            data_compress["e"],
            predict_dir,
            mode,
        )

    if mode == "predict_xyz":
        for id_, mean_y_predict_, std_y_predict_ in tqdm.tqdm(
            zip(data_compress["ids"], prob_mean, prob_var)
        ):
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xyz_mean(
                    f, mean_y_predict_.detach().numpy(), std_y_predict_.detach().numpy()
                )

    elif mode == "predict_xanes":
        for id_, mean_y_predict_, std_y_predict_ in tqdm.tqdm(
            zip(data_compress["ids"], prob_mean, prob_var)
        ):
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes_mean(
                    f,
                    XANES(data_compress["e"], mean_y_predict_.detach().numpy()),
                    std_y_predict_.detach().numpy(),
                )


def montecarlo_dropout_ae(
    model, input_data, n_mc, data_compress, predict_dir, mode, plot_save
):
    from plot import plot_mc_ae_predict

    model.train()

    prob_output = []
    prob_recon = []

    input_data = torch.from_numpy(input_data)
    input_data = input_data.float()

    print("Running Monte-carlo dropout")
    for t in range(n_mc):
        recon, output = model(input_data)
        prob_output.append(output)
        prob_recon.append(recon)

    mean_output = torch.mean(torch.stack(prob_output), dim=0)
    var_output = torch.std(torch.stack(prob_output), dim=0)

    mean_recon = torch.mean(torch.stack(prob_recon), dim=0)
    var_recon = torch.std(torch.stack(prob_recon), dim=0)

    print(
        "MSE x to x prob : ",
        mean_squared_error(data_compress["x"], mean_recon.detach().numpy()),
    )
    print(
        "MSE y to y prob : ",
        mean_squared_error(data_compress["y"], mean_output.detach().numpy()),
    )
    # confidence interval
    if plot_save:
        plot_mc_ae_predict(
            data_compress["ids"],
            data_compress["y"],
            data_compress["y_predict"],
            data_compress["x"],
            data_compress["x_recon"],
            mean_output.detach().numpy(),
            var_output.detach().numpy(),
            mean_recon.detach().numpy(),
            var_recon.detach().numpy(),
            data_compress["e"],
            predict_dir,
            mode,
        )


def montecarlo_dropout_aegan(model, x, y, n_mc):
    model.train()

    if x is not None and y is not None:
        prob_x_pred = []
        prob_y_pred = []
        prob_x_recon = []
        prob_y_recon = []
    elif x is not None and y is None:
        prob_x_recon = []
    elif y is not None and x is None:
        prob_y_recon = []

    print("Running Monte-carlo dropout")
    for t in range(n_mc):
        if x is not None and y is not None:
            prob_x_pred.append(model.predict_structure(y))
            prob_y_pred.append(model.predict_spectrum(x))
            prob_x_recon.append(model.reconstruct_structure(x))
            prob_y_recon.append(model.reconstruct_spectrum(y))
        elif x is not None and y is None:
            prob_x_recon.append(model.reconstruct_structure(x))
        elif y is not None and x is None:
            prob_y_recon.append(model.reconstruct_spectrum(y))

    if x is not None and y is not None:
        mean_x_pred = torch.mean(torch.stack(prob_x_pred), dim=0)
        var_x_pred = torch.std(torch.stack(prob_x_pred), dim=0)
        print(
            "MSE x to x pred : ",
            mean_squared_error(x, mean_x_pred.detach().numpy()),
        )
        mean_y_pred = torch.mean(torch.stack(prob_y_pred), dim=0)
        var_y_pred = torch.std(torch.stack(prob_y_pred), dim=0)
        print(
            "MSE y to y pred : ",
            mean_squared_error(y, mean_y_pred.detach().numpy()),
        )
        mean_x_recon = torch.mean(torch.stack(prob_x_recon), dim=0)
        var_x_recon = torch.std(torch.stack(prob_x_recon), dim=0)
        print(
            "MSE x to x recon : ",
            mean_squared_error(x, mean_x_recon.detach().numpy()),
        )
        mean_y_recon = torch.mean(torch.stack(prob_y_recon), dim=0)
        var_y_recon = torch.std(torch.stack(prob_y_recon), dim=0)
        print(
            "MSE y to y recon : ",
            mean_squared_error(y, mean_y_recon.detach().numpy()),
        )
    elif x is not None and y is None:
        mean_x_recon = torch.mean(torch.stack(prob_x_recon), dim=0)
        var_x_recon = torch.std(torch.stack(prob_x_recon), dim=0)
        print(
            "MSE x to x recon : ",
            mean_squared_error(x, mean_x_recon.detach().numpy()),
        )
    elif y is not None and x is None:
        mean_y_recon = torch.mean(torch.stack(prob_y_recon), dim=0)
        var_y_recon = torch.std(torch.stack(prob_y_recon), dim=0)
        print(
            "MSE y to y recon : ",
            mean_squared_error(y, mean_y_recon.detach().numpy()),
        )

