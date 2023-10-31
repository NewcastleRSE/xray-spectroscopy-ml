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

import random
import numpy as np


# DATA AUGMENTATION
def data_augment(data_params, xyz_data, xanes_data, index, n_x_features, n_y_features):
    n_samples = len(index)
    n_aug_samples = np.multiply(n_samples, data_params["augment_mult"]) - n_samples

    if data_params["type"].lower() == "random_noise":
        # augment data as random data point + noise

        rand = random.choices(range(n_samples), k=n_aug_samples)
        noise1 = np.random.normal(
            data_params["normal_mean"],
            data_params["normal_sd"],
            (n_aug_samples, n_x_features),
        )
        noise2 = np.random.normal(
            data_params["normal_mean"],
            data_params["normal_sd"],
            (n_aug_samples, n_y_features),
        )

        data1 = xyz_data[rand, :] + noise1
        data2 = xanes_data[rand, :] + noise2

    elif data_params["type"].lower() == "random_combination":
        rand1 = random.choices(range(n_samples), k=n_aug_samples)
        rand2 = random.choices(range(n_samples), k=n_aug_samples)

        data1 = 0.5 * (xyz_data[rand1, :] + xyz_data[rand2, :])
        data2 = 0.5 * (xanes_data[rand1, :] + xanes_data[rand2, :])

    else:
        raise ValueError("augment type not found")

    xyz_data = np.vstack((xyz_data, data1))
    xanes_data = np.vstack((xanes_data, data2))

    return xyz_data, xanes_data
