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

import numpy as np


def fourier_transform(x, concat):
    """
    Transform xanes spectra using Fourier
    """

    y = np.hstack((x, x[:, ::-1]))
    f = np.fft.fft(y)
    z = f.real

    # Combine features
    if concat:
        z = np.concatenate((x, z), axis=1)

    return z


def inverse_fourier_transform(z, concat):
    """
    Get inverse of fourier transformed data
    """
    # Decompose features
    if concat:
        z = z[:, z.shape[1] // 3 :]

    iz = np.fft.ifft(z).real[:, : z.shape[1] // 2]

    return iz
