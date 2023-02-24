import torch
import numpy as np


def average(lst):
    for lstNum in range(len(lst)):
        print(lstNum)
        for sublistItem in range(len(lst[lstNum])):
            lst[lstNum] / lst[sublistItem]  # <-- ??
    print(type(lst))
    return lst


def y_predict_dim(y_predict, ids, model_dir):
    if y_predict.ndim == 1:
        if len(ids) == 1:
            y_predict = y_predict.reshape(-1, y_predict.size)
        else:
            y_predict = y_predict.reshape(y_predict.size, -1)
    print(">> ...predicted Y data!\n")

    with open(model_dir / "dataset.npz", "rb") as f:
        e = np.load(f)["e"]

    return y_predict, e.flatten()


def predict_xyz(xanes_data, model):
    print("predict xyz structure")
    xanes = torch.from_numpy(xanes_data)
    xanes = xanes.float()

    pred_xyz = model(xanes)

    return pred_xyz


def predict_xanes(xyz_data, model):
    print("predict xanes spectrum")
    xyz = torch.from_numpy(xyz_data)
    xyz = xyz.float()

    pred_xanes = model(xyz)

    return pred_xanes
