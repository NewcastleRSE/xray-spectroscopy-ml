"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

###############################################################################
################################## CLASSES ####################################
###############################################################################


class XANES:
    def __init__(
        self, e: np.ndarray, m: np.ndarray, e0: float = None, info: dict = None
    ):
        """
        Args:
            e (np.ndarray; 1D): an array of energy (`e`; eV) values
            m (np.ndarray; 1D): an array of intensity (`m`; arbitrary) values
            e0 (float, optional): the X-ray absorption edge (`e0`; eV) energy.
                If None, an attempt is made to determine `e0` from the maximum
                derivative of `m` with `get_e0()`.
                Defaults to None.
            info (dict, optional): a dictionary of key/val pairs that can be
                used to store extra information about the XANES spectrum as a
                tag.
                Defaults to None.

        Raises:
            ValueError: if the `e` and `m` arrays are not the same length.
        """

        if len(e) == len(m):
            self._e = e
            self._m = m
        else:
            raise ValueError(
                "the energy (`e`) and XANES spectral intensity "
                "(`m`) arrays are not the same length"
            )

        if e0 is not None:
            self._e0 = e0
        else:
            self._e0 = self.estimate_e0()

        if info is not None:
            self.info = info
        else:
            self.info = {}

    def estimate_e0(self) -> float:
        """
        Estimates the X-ray absorption edge (`e0`; eV) energy as the energy
        `e` where the derivative of `m` is largest.

        Returns:
            float: the X-ray absorption edge (`e0`; eV) energy.
        """

        return self._e[np.argmax(np.gradient(self._m))]

    def scale(self, fit_limits: tuple = (100.0, 400.0), flatten: bool = True):
        """
        Scales the XANES spectrum using the 'edge-step' approach: fitting a
        2nd-order (quadratic) polynomial, `fit`, to the post-edge (where `e`
        >= `e0`; eV), determining the 'edge-step', `fit(e0)`, and scaling
        `m` by dividing through by this value. `m` can also be flattened;
        in this case, the post-edge is levelled off to ca. 1.0 by adding
        (1.0 - `fit(e0)`) to `m` where `e` >= `e0`.

        Args:
            fit_limits (tuple, optional): lower and upper limits (in eV
                relative to the X-ray absorption edge; `e0`) defining the `e`
                window over which the 2nd-order (quadratic) polynomial, `fit`,
                is determined.
                Defaults to (100.0, 400.0).
            flatten (bool, optional): toggles flattening of the post-edge by
                adding (1.0 - `fit(e0)`) to `m` where `e` >= `e0`.
                Defaults to True.
        """

        e_rel = self._e - self._e0
        e_rel_min, e_rel_max = fit_limits

        fit_window = (e_rel >= e_rel_min) & (e_rel <= e_rel_max)

        fit = np.polynomial.Polynomial.fit(
            self._e[fit_window], self._m[fit_window], deg=2
        )

        self._m /= fit(self._e0)

        if flatten:
            self._m[self._e >= self._e0] += 1.0 - (
                fit(self._e)[self._e >= self._e0] / fit(self._e0)
            )

        return self

    def convolve(
        self,
        conv_type: str = "fixed_width",
        width: float = 2.0,
        ef: float = -5.0,
        ec: float = 30.0,
        el: float = 30.0,
        width_max: float = 15.0,
    ):
        """
        Convolves the XANES spectrum with either a fixed-width (`width`; eV)
        Lorentzian function (if `conv_type` == 'fixed_width') or energy-
        dependent variable-width Lorentzian function where the width is derived
        from either the Seah-Dench (if `conv_type` == 'seah_dench_model') or
        arctangent (if `conv_type` == 'arctangent_model') convolution models.
        `m` is projected from `e` onto an auxilliary energy scale `e_aux` via
        linear interpolation before convolution; the spacing of the energy
        gridpoints in `e_aux` is equal to the smallest spacing of the energy
        gridpoints in `e`, and `e_aux` is padded at either end.

        Args:
            conv_type (str, optional): type of convolution; options are
                'fixed_width', 'seah_dench_model', and 'arctangent_model'.
                Defaults to 'fixed_width'.
            width (float, optional): width (in eV) of the Lorentzian function
                used in the convolution if `conv_type` == 'fixed_width'; if
                `conv_type` == 'seah_dench_model' or 'arctangent_model', this
                is the initial width of the (energy-dependent) Lorentzian.
                Defaults to 2.0.
            ef (float, optional): the Fermi energy (in eV, relative to `e0`);
                cross-sectional contributions from the occupied states below
                the Fermi energy are removed.
                Defaults to -5.0.
            ec (float, optional): the centre of the arctangent function (in eV,
                relative to `e0`); used if `conv_type` == 'arctangent_model'.
                Defaults to 30.0.
            el (float, optional): the width of the arctangent function (in eV);
                used if `conv_type` == 'arctangent_model'.
                Defaults to 30.0.
            width_max (float, optional): the maximum width (in eV) used in the
                convolution; used if `conv_type` == 'seah_dench_model' or
                'arctangent_model'.
                Defaults to 15.0.

        Raises:
            ValueError: _description_
        """

        de = np.min(np.diff(self._e))

        pad = de * int((50.0 * width) / de)

        e_aux = np.linspace(
            np.min(self._e) - pad,
            np.max(self._e) + pad,
            int((np.ptp(self._e) + (2.0 * pad)) / de) + 1,
        )

        if conv_type == "fixed_width":
            pass
        elif conv_type == "seah_dench_model":
            width = _calc_seah_dench_conv_width(
                e_rel=e_aux - self._e0, width=width, ef=ef, width_max=width_max
            )
        elif conv_type == "arctangent_model":
            width = _calc_arctangent_conv_width(
                e_rel=e_aux - self._e0,
                width=width,
                ef=ef,
                ec=ec,
                el=el,
                width_max=width_max,
            )
        else:
            raise ValueError(
                "the convolution type is not recognised; try"
                "`fixed_width` or `arctangent`"
            )

        # remove cross-sectional contributions to `m` below `ef`
        self._m[self._e < (self._e0 + ef)] = 0.0

        # project `m` onto the auxilliary energy scale `e_aux`
        m_aux = np.interp(e_aux, self._e, self._m)

        e_, e0_ = np.meshgrid(e_aux, e_aux)
        conv_filter = _lorentzian(e_, e0_, width)

        # convolve `m_aux` with the convolution filter `conv_filter`
        m_aux = np.sum(conv_filter * m_aux, axis=1)

        # project `m_aux` onto the original energy scale `e`
        self._m = np.interp(self._e, e_aux, m_aux)

        return self

    @property
    def e(self) -> float:
        return self._e

    @property
    def m(self) -> float:
        return self._m

    @property
    def e0(self) -> float:
        return self._e0

    @property
    def spectrum(self) -> tuple:
        return (self._e, self._m)


def _lorentzian(x: np.ndarray, x0: float, width: float):
    # returns the `y` values for a Lorentzian function defined over `x` with a
    # centre `x0` and a width `width`

    return width * (0.5 / ((x - x0) ** 2 + (0.5 * width) ** 2))


def _calc_seah_dench_conv_width(
    e_rel: np.ndarray,
    width: float = 2.0,
    ef: float = -5.0,
    a: float = 1.0,
    width_max: float = 15.0,
) -> np.ndarray:
    # returns the widths for an energy-dependent Lorentzian under the
    # Seah-Dench convolution model; evaluated over `e_rel` (the energy relative
    # to the X-ray absorption edge `e0` in eV) where `width` is the initial
    # state width in eV, `width_max` is the final state width in eV, `ef` is
    # the Fermi energy in eV, and `a` is the Seah-Dench (pre-)factor

    e_ = e_rel - ef

    g = width + ((a * width_max * e_) / (width_max + (a * e_)))

    return g


def _calc_arctangent_conv_width(
    e_rel: np.ndarray,
    width: float = 2.0,
    ef: float = -5.0,
    ec: float = 30.0,
    el: float = 30.0,
    width_max: float = 15.0,
) -> np.ndarray:
    # returns the widths for an energy-dependent Lorentzian under the
    # arctangent convolution model; evaluated over `e_rel` (the energy relative
    # to the X-ray absorption edge `e0` in eV) where `width` is the initial
    # state width in eV, `width_max` is the final state width in eV, `ef` is
    # the Fermi energy in eV, `ec` is the centre of the arctangent in eV
    # relative to `e0`, and `el` is the width of the arctangent in eV

    e_ = (e_rel - ef) / ec

    with np.errstate(divide="ignore"):
        arctan = (np.pi / 3.0) * (width_max / el) * (e_ - (1.0 / e_**2))

    g = width + (width_max * ((1.0 / 2.0) + (1.0 / np.pi) * np.arctan(arctan)))

    return g
