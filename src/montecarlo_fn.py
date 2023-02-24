import torch
from sklearn.metrics import mean_squared_error


def montecarlo_dropout(model, input_data, n_mc, data_compress, predict_dir, mode):
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
    # confidence interval
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


def montecarlo_dropout_ae(model, input_data, n_mc, data_compress, predict_dir, mode):
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
