import seaborn as sns
import matplotlib.pyplot as plt
import tqdm as tqdm
import numpy as np

from inout import save_xanes
from spectrum.xanes import XANES

from sklearn.metrics.pairwise import cosine_similarity


def plot_predict(ids, y, y_predict, e, predict_dir, mode):
    total_y = []
    total_y_pred = []
    for id_, y_predict_, y_ in tqdm.tqdm(zip(ids, y_predict, y)):
        sns.set()
        plt.figure()
        plt.plot(y_predict_.detach().numpy(), label="prediction")
        plt.plot(y_, label="target")
        plt.legend(loc="upper right")
        total_y.append(y_)
        total_y_pred.append(y_predict_.detach().numpy())

        if mode == "predict_xanes":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))
                plt.savefig(predict_dir / f"{id_}.pdf")

        elif mode == "predict_xyz":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))
                plt.savefig(predict_dir / f"{id_}.pdf")

        plt.close()

    print(">> saving Y data predictions...")

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)

    # plotting the average loss
    sns.set_style("dark")
    plt.figure()

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)
    plt.plot(mean_y, label="target")

    plt.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)
    plt.plot(mean_y_pred, label="prediction")
    plt.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    plt.legend(loc="best")
    plt.grid()
    plt.savefig(predict_dir / "avg_plot.pdf")

    plt.show()


def plot_ae_predict(ids, y, y_predict, x, x_recon, e, predict_dir, mode):
    total_y = []
    total_y_pred = []
    total_x = []
    total_x_recon = []

    for id_, y_predict_, y_, x_recon_, x_ in tqdm.tqdm(
        zip(ids, y_predict, y, x_recon, x)
    ):
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(y_predict_.detach().numpy(), label="prediction")
        ax1.set_title("prediction")
        ax1.plot(y_, label="target")
        ax1.legend(loc="upper right")

        ax2.plot(x_recon_.detach().numpy(), label="prediction")
        ax2.set_title("reconstruction")
        ax2.plot(x_, label="target")
        ax2.legend(loc="upper right")
        # print(type(x_))
        total_y.append(y_)
        total_y_pred.append(y_predict_.detach().numpy())

        # total_x.append(x_.detach().numpy())
        total_x.append(x_)
        total_x_recon.append(x_recon_.detach().numpy())
        if mode == "predict_xanes":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))
                plt.savefig(predict_dir / f"{id_}.pdf")

        elif mode == "predict_xyz":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))
                plt.savefig(predict_dir / f"{id_}.pdf")

        fig.clf()
        plt.close(fig)

    print(">> saving Y data predictions...")

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)
    total_x = np.asarray(total_x)
    total_x_recon = np.asarray(total_x_recon)

    # plotting the average loss
    sns.set_style("dark")
    fig, (ax1, ax2) = plt.subplots(2)

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)

    ax1.plot(mean_y, label="target")
    ax1.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)

    ax1.plot(mean_y_pred, label="prediction")
    ax1.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    ax1.legend(loc="best")
    ax1.grid()

    mean_x = np.mean(total_x, axis=0)
    stddev_x = np.std(total_x, axis=0)

    ax2.plot(mean_x, label="target")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    mean_x = np.mean(total_x_recon, axis=0)
    stddev_x = np.std(total_x_recon, axis=0)

    ax2.plot(mean_x, label="reconstruction")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    ax2.legend(loc="best")
    ax2.grid()

    plt.savefig(predict_dir / "avg_plot.pdf")

    plt.show()
    fig.clf()
    plt.close(fig)


def plot_running_aegan(losses, model_dir):
    print(">> Plotting running losses...")

    sns.set()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 10))

    ax1.plot(losses["train_loss"], label="Total training loss", color=cycle[0])
    ax1.legend(loc="upper right")

    ax2.plot(
        losses["loss_x_recon"],
        label="Training Structure Reconstruction",
        color=cycle[1],
    )
    ax2.plot(
        losses["loss_x_pred"], label="Training Structure Prediction", color=cycle[2]
    )
    ax2.legend(loc="upper right")

    ax3.plot(
        losses["loss_y_recon"], label="Training Spectrum Reconstruction", color=cycle[3]
    )
    ax3.plot(
        losses["loss_y_pred"], label="Training Spectrum Prediction", color=cycle[4]
    )
    ax3.legend(loc="upper right")

    # # with open(predict_dir / f'{id_}.txt', 'w') as f:
    #     # save_xanes(f, XANES(e, y_predict_.detach().numpy()))
    plt.savefig(f"{model_dir}/training-mse-loss.pdf")
    fig.clf()
    plt.close(fig)


def plot_aegan_predict(ids, x, y, x_recon, y_recon, x_pred, y_pred, plots_dir):
    for id_, x_, y_, x_recon_, y_recon_, x_pred_, y_pred_ in tqdm.tqdm(
        zip(ids, x, y, x_recon, y_recon, x_pred, y_pred)
    ):
        sns.set()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 20))

        ax1.plot(x_recon_, label="Reconstruction")
        ax1.set_title(f"Structure Reconstruction")
        ax1.plot(x_, label="target")
        ax1.legend(loc="upper left")

        ax2.plot(y_recon_, label="Reconstruction")
        ax2.set_title(f"Spectrum Reconstruction")
        ax2.plot(y_, label="target")
        ax2.legend(loc="upper left")

        ax3.plot(y_pred_, label="Prediction")
        ax3.set_title(f"Spectrum Prediction")
        ax3.plot(y_, label="target")
        ax3.legend(loc="upper left")

        ax4.plot(x_pred_, label="Prediction")
        ax4.set_title(f"Structure Prediction")
        ax4.plot(x_, label="target")
        ax4.legend(loc="upper left")

        plt.savefig(plots_dir / f"{id_}.pdf")
        fig.clf()
        plt.close(fig)


def plot_aegan_spectrum(ids, x, x_recon, y_pred, plots_dir):
    for id_, x_, x_recon_, y_pred_ in tqdm.tqdm(zip(ids, x, x_recon, y_pred)):
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

        ax1.plot(x_recon_, label="Reconstruction")
        ax1.set_title(f"Structure Reconstruction")
        ax1.plot(x_, label="target")
        ax1.legend(loc="upper left")

        ax2.plot(y_pred_, label="Prediction")
        ax2.set_title(f"Spectrum Prediction")
        ax2.legend(loc="upper left")

        plt.savefig(plots_dir / f"{id_}.pdf")
        fig.clf()
        plt.close(fig)


def plot_aegan_structure(ids, y, y_recon, x_pred, plots_dir):
    for id_, y_, y_recon_, x_pred_ in tqdm.tqdm(zip(ids, y, y_recon, x_pred)):
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))

        ax1.plot(y_recon_, label="Reconstruction")
        ax1.set_title(f"Sprectum Reconstruction")
        ax1.plot(y_, label="target")
        ax1.legend(loc="upper left")

        ax2.plot(x_pred_, label="Prediction")
        ax2.set_title(f"Structure Prediction")
        ax2.legend(loc="upper left")

        plt.savefig(plots_dir / f"{id_}.pdf")
        fig.clf()
        plt.close(fig)


def plot_cosine_similarity(x, y, x_recon, y_recon, x_pred, y_pred, analysis_dir):
    cosine_x_x_pred = np.diagonal(cosine_similarity(x, x_pred))
    cosine_y_y_pred = np.diagonal(cosine_similarity(y, y_pred))
    cosine_x_x_recon = np.diagonal(cosine_similarity(x, x_recon))
    cosine_y_y_recon = np.diagonal(cosine_similarity(y, y_recon))

    sns.set()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 15))
    ax1.plot(cosine_x_x_recon, cosine_y_y_pred, "o", color=cycle[0])
    ax1.set(xlabel="Reconstructed Structure", ylabel="Predicted Spectrum")
    ax2.plot(cosine_y_y_recon, cosine_x_x_pred, "o", color=cycle[1])
    ax2.set(xlabel="Reconstructed Spectrum", ylabel="Predicted Structure")
    ax3.plot(
        cosine_x_x_recon + cosine_y_y_recon,
        cosine_x_x_pred + cosine_y_y_pred,
        "o",
        color=cycle[2],
    )
    ax3.set(xlabel="Reconstruction", ylabel="Prediction")
    plt.savefig(f"{analysis_dir}/cosine_similarity.pdf")
    fig.clf()
    plt.close(fig)


def plot_mc_predict(ids, y, y_predict, prob_mean, prob_var, e, predict_dir, mode):
    total_y = []
    total_y_pred = []
    for id_, y_predict_, y_, prob_mean_, prob_var_ in tqdm.tqdm(
        zip(ids, y_predict, y, prob_mean, prob_var)
    ):
        sns.set()
        plt.figure()
        plt.plot(y_predict_.detach().numpy(), label="prediction")
        plt.plot(y_, label="target")
        plt.plot(prob_mean_, label="monte_carlo")

        plt.fill_between(
            np.arange(prob_mean_.shape[0]),
            prob_mean_ + prob_var_,
            prob_mean_ - prob_var_,
            alpha=0.4,
            linewidth=0,
        )
        plt.legend(loc="upper right")
        total_y.append(y_)
        total_y_pred.append(y_predict_.detach().numpy())

        if mode == "predict_xanes":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))
                plt.savefig(predict_dir / f"{id_}.pdf")

        elif mode == "predict_xyz":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))
                plt.savefig(predict_dir / f"{id_}.pdf")

        plt.close()

    print(">> saving Y data predictions...")

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)

    # plotting the average loss
    sns.set_style("dark")
    plt.figure()

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)
    plt.plot(mean_y, label="target")

    plt.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)
    plt.plot(mean_y_pred, label="prediction")
    plt.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    plt.legend(loc="best")
    plt.grid()
    plt.savefig(predict_dir / "avg_plot.pdf")

    plt.show()


def plot_mc_ae_predict(
    ids,
    y,
    y_predict,
    x,
    x_recon,
    mean_output,
    var_output,
    mean_recon,
    var_recon,
    e,
    predict_dir,
    mode,
):
    total_y = []
    total_y_pred = []
    total_x = []
    total_x_recon = []
    for (
        id_,
        y_predict_,
        y_,
        x_recon_,
        x_,
        mean_output_,
        var_output_,
        mean_recon_,
        var_recon_,
    ) in tqdm.tqdm(
        zip(
            ids,
            y_predict,
            y,
            x_recon,
            x,
            mean_output,
            var_output,
            mean_recon,
            var_recon,
        )
    ):
        sns.set()
        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(y_predict_.detach().numpy(), label="prediction")
        ax1.plot(y_, label="target")
        ax1.plot(mean_output_, label="monte_carlo")
        ax1.set_title("prediction")
        ax1.legend(loc="upper right")

        ax1.fill_between(
            np.arange(mean_output_.shape[0]),
            mean_output_ + var_output_,
            mean_output_ - var_output_,
            alpha=0.4,
            linewidth=0,
        )

        ax2.plot(x_recon_.detach().numpy(), label="reconstruction")
        ax2.plot(x_, label="target")
        ax2.plot(mean_recon_, label="monte_carlo")
        ax2.set_title("prediction")
        ax2.legend(loc="upper right")

        ax2.fill_between(
            np.arange(mean_recon_.shape[0]),
            mean_recon_ + var_recon_,
            mean_recon_ - var_recon_,
            alpha=0.4,
            linewidth=0,
        )

        plt.legend(loc="upper right")

        if mode == "predict_xanes":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                save_xanes(f, XANES(e, y_predict_.detach().numpy()))
                plt.savefig(predict_dir / f"{id_}.pdf")

        elif mode == "predict_xyz":
            with open(predict_dir / f"{id_}.txt", "w") as f:
                f.write("\n".join(map(str, y_predict_.detach().numpy())))
                plt.savefig(predict_dir / f"{id_}.pdf")

        fig.clf()
        plt.close(fig)

    print(">> saving Y data predictions...")

    total_y.append(y_)
    total_y_pred.append(y_predict_.detach().numpy())

    total_x.append(x_)
    total_x_recon.append(x_recon_.detach().numpy())

    total_y = np.asarray(total_y)
    total_y_pred = np.asarray(total_y_pred)
    total_x = np.asarray(total_x)
    total_x_recon = np.asarray(total_x_recon)

    # plotting the average loss
    sns.set_style("dark")
    fig, (ax1, ax2) = plt.subplots(2)

    mean_y = np.mean(total_y, axis=0)
    stddev_y = np.std(total_y, axis=0)

    ax1.plot(mean_y, label="target")
    ax1.fill_between(
        np.arange(mean_y.shape[0]),
        mean_y + stddev_y,
        mean_y - stddev_y,
        alpha=0.4,
        linewidth=0,
    )

    mean_y_pred = np.mean(total_y_pred, axis=0)
    stddev_y_pred = np.std(total_y_pred, axis=0)

    ax1.plot(mean_y_pred, label="prediction")
    ax1.fill_between(
        np.arange(mean_y_pred.shape[0]),
        mean_y_pred + stddev_y_pred,
        mean_y_pred - stddev_y_pred,
        alpha=0.4,
        linewidth=0,
    )

    ax1.legend(loc="best")
    ax1.grid()

    mean_x = np.mean(total_x, axis=0)
    stddev_x = np.std(total_x, axis=0)

    ax2.plot(mean_x, label="target")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    mean_x = np.mean(total_x_recon, axis=0)
    stddev_x = np.std(total_x_recon, axis=0)

    ax2.plot(mean_x, label="reconstruction")
    ax2.fill_between(
        np.arange(mean_x.shape[0]),
        mean_x + stddev_x,
        mean_x - stddev_x,
        alpha=0.4,
        linewidth=0,
    )

    ax2.legend(loc="best")
    ax2.grid()

    plt.savefig(predict_dir / "avg_plot.pdf")

    plt.show()
    fig.clf()
    plt.close(fig)
