"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import numpy as np


def average(lst):
    for lstNum in range(len(lst)):
        print(lstNum)
        for sublistItem in range(len(lst[lstNum])):
            lst[lstNum] / lst[sublistItem]  # <-- ??
    print(type(lst))
    return lst


def y_predict_dim(y_predict, ids):
    if y_predict.ndim == 1:
        if len(ids) == 1:
            y_predict = y_predict.reshape(-1, y_predict.size)
        else:
            y_predict = y_predict.reshape(y_predict.size, -1)
    print(">> ...predicted Y data!\n")

    return y_predict


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
