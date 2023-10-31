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

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

from ase import Atoms
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

from xanesnet.descriptor.vector_descriptor import VectorDescriptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class MSR(VectorDescriptor):
    """
    A class for transforming a molecular system into a multiple scattering
    descriptor. MSRs encode the local geometry
    around an absorption site in a manner reminescent of the path expansion in
    multiple scattering theory.
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 8.0,
        n_s2: int = 0,
        n_s3: int = 0,
        n_s4: int = 0,
        n_s5: int = 0,
        use_charge=False,
        use_spin=False,
    ):
        """
        Args:
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            n_s2 (int, optional): Two body terms to use for encoding.
                Defaults to 0.
            n_s3 (int, optional): Three body terms to use for encoding.
                Defaults to 0.
            n_s4 (int, optional): Four body terms to use for encoding.
                Defaults to 0.
            n_s5 (int, optional): Five body terms to use for encoding.
                Defaults to 0.
            use_charge (bool): If True, includes an additional element in the
                vector descriptor for the charge state of the complex.
                Defaults to False.
            use_spin (bool): If True, includes an additional element in the
                vector descriptor for the spin state of the complex.
                Defaults to False.
        """

        super().__init__(r_min, r_max, use_charge, use_spin)

        self.n_s2 = n_s2
        self.n_s3 = n_s3
        self.n_s4 = n_s4
        self.n_s5 = n_s5

        if self.n_s2:
            self.s2_transformer = S2SymmetryFunctionTransformer(
                self.n_s2,
                r_min=self.r_min,
                r_max=self.r_max,
            )

        if self.n_s3:
            self.s3_transformer = S3SymmetryFunctionTransformer(
                self.n_s3,
                r_min=self.r_min,
                r_max=self.r_max,
            )

        if self.n_s4:
            self.s4_transformer = S4SymmetryFunctionTransformer(
                self.n_s4,
                r_min=self.r_min,
                r_max=self.r_max,
            )

        if self.n_s5:
            self.s5_transformer = S5SymmetryFunctionTransformer(
                self.n_s4,
                r_min=self.r_min,
                r_max=self.r_max,
            )

    def transform(self, system: Atoms) -> np.ndarray:
        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]

        ij = np.array([[0, j] for j in range(1, len(system))], dtype="uint16")

        if self.n_s3:
            ijk = np.array(
                [
                    [0, j, k]
                    for j in range(1, len(system))
                    for k in range(1, len(system))
                    if k > j
                ],
                dtype="uint16",
            )

        if self.n_s4:
            ijkl = np.array(
                [
                    [0, j, k, l]
                    for j in range(1, len(system))
                    for k in range(1, len(system))
                    if k > j
                    for l in range(1, len(system))
                    if l > k
                ],
                dtype="uint16",
            )

        if self.n_s5:
            ijklm = np.array(
                [
                    [0, j, k, l, m]
                    for j in range(1, len(system))
                    for k in range(1, len(system))
                    if k > j
                    for l in range(1, len(system))
                    if l > k
                    for m in range(1, len(system))
                    if m > l
                ],
                dtype="uint16",
            )

        rij = system.get_distances(ij[:, 0], ij[:, 1])

        msr = []

        if self.n_s2:
            zj = system.get_atomic_numbers()[ij[:, 1]]
            zj = 0.1 * zj
            rij = system.get_distances(ij[:, 0], ij[:, 1])
            s2 = self.s2_transformer.transform(zj, rij)
            msr = np.append(msr, s2)

        if self.n_s3:
            zj = system.get_atomic_numbers()[ijk[:, 1]]
            zk = system.get_atomic_numbers()[ijk[:, 2]]
            zj = 0.1 * zj
            zk = 0.1 * zk
            rij = system.get_distances(ijk[:, 0], ijk[:, 1])
            rik = system.get_distances(ijk[:, 1], ijk[:, 2])
            rjk = system.get_distances(ijk[:, 0], ijk[:, 2])
            rijk = rij + rik
            aijk = np.radians(system.get_angles(ijk))
            s3 = self.s3_transformer.transform(zj, zk, rijk, aijk)
            msr = np.append(msr, s3)

        if self.n_s4:
            zj = system.get_atomic_numbers()[ijkl[:, 1]]
            zk = system.get_atomic_numbers()[ijkl[:, 2]]
            zl = system.get_atomic_numbers()[ijkl[:, 3]]
            zj = 0.1 * zj
            zk = 0.1 * zk
            zl = 0.1 * zl
            rij = system.get_distances(ijkl[:, 0], ijkl[:, 1])
            rik = system.get_distances(ijkl[:, 1], ijkl[:, 2])
            rkl = system.get_distances(ijkl[:, 2], ijkl[:, 3])
            rijkl = rij + rik + rkl
            aijk = np.radians(system.get_angles(ijkl[:, :3]))
            ajkl = np.radians(system.get_angles(ijkl[:, 1:]))
            s4 = self.s4_transformer.transform(zj, zk, zl, rijkl, aijk, ajkl)
            msr = np.append(msr, s4)

        if self.n_s5:
            zj = system.get_atomic_numbers()[ijklm[:, 1]]
            zk = system.get_atomic_numbers()[ijklm[:, 2]]
            zl = system.get_atomic_numbers()[ijklm[:, 3]]
            zm = system.get_atomic_numbers()[ijklm[:, 4]]
            zj = 0.1 * zj
            zk = 0.1 * zk
            zl = 0.1 * zl
            zm = 0.1 * zm
            rij = system.get_distances(ijklm[:, 0], ijklm[:, 1])
            rik = system.get_distances(ijklm[:, 1], ijklm[:, 2])
            rkl = system.get_distances(ijklm[:, 2], ijklm[:, 3])
            rlm = system.get_distances(ijklm[:, 3], ijklm[:, 4])
            rijklm = rij + rik + rkl + rlm
            aijk = np.radians(system.get_angles(ijklm[:, 0:3]))
            ajkl = np.radians(system.get_angles(ijklm[:, 1:4]))
            aklm = np.radians(system.get_angles(ijklm[:, 2:5]))
            s5 = self.s5_transformer.transform(zj, zk, zl, zm, rijklm, aijk, ajkl, aklm)
            msr = np.append(msr, s5)

        if self.use_spin:
            msr = np.append(system.info["S"], msr)

        if self.use_charge:
            msr = np.append(system.info["q"], msr)

        return msr

    def get_number_of_features(self) -> int:
        return int(
            self.n_s2
            + self.n_s3
            + self.n_s4
            + self.n_s5
            + self.use_charge
            + self.use_spin
        )

    def process(self, atoms: Atoms):
        return self.transform(atoms)


class SymmetryFunctionTransformer(ABC):
    """
    An abstract base class for generating multiple scattering vector.
    """

    def __init__(self, n: int, r_min: float, r_max: float):
        """
        Args:
            n (int): The number of functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
        """

        self.n = n
        self.r_min = r_min
        self.r_max = r_max

        r_aux = np.linspace(self.r_min + 0.5, self.r_max - 0.5, self.n)
        dr = np.diff(r_aux)[0]
        self.h = np.array([1.0 / (2.0 * (dr**2)) for _ in r_aux])
        self.m = np.array([i for i in r_aux])

        theta_aux = np.linspace(0, np.pi, 18)
        dtheta = np.diff(theta_aux)[0]
        self.th = np.array([1.0 / (2.0 * (dtheta**2)) for _ in theta_aux])
        self.tm = np.array([i for i in theta_aux])

    @abstractmethod
    def transform(self, *args) -> np.ndarray:
        """
        Encodes structural information (`*args`) using symmetry functions.

        Returns:
            np.ndarray: A symmetry function vector.
        """

        pass


class S2SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating two body (S2) terms.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
    ):
        """
        Args:
            n (int): The number of two body (S2) terms for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
        """

        super().__init__(n, r_min=r_min, r_max=r_max)

    def transform(self, zj: np.ndarray, rij: np.ndarray) -> np.ndarray:
        s2 = np.full(self.n, np.nan)

        cutoff_ij = cosine_cutoff(rij, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            s2[i] = np.sum(zj * gaussian(rij, h, m) * cutoff_ij)
            i += 1

        return s2


class S3SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating three body (S3) terms.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
    ):
        """
        Args:
            n (int): The number of three body (S3) terms for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
        """

        super().__init__(n, r_min=r_min, r_max=r_max)

        n_ = self.n
        self.h = self.h[:n_]
        self.m = self.m[:n_]

    def transform(
        self,
        zj: np.ndarray,
        zk: np.ndarray,
        rijk: np.ndarray,
        aijk: np.ndarray,
    ) -> np.ndarray:
        s3 = np.full(self.n, np.nan)

        cutoff_ijk = cosine_cutoff(rijk, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            s3[i] = np.sum(
                zj * zk * (np.abs(np.cos(aijk))) * gaussian(rijk, h, m) * cutoff_ijk
            )
            i += 1

        return s3


class S4SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating four body (S4) terms.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
    ):
        """
        Args:
            n (int): The number of four body (S4) terms for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
        """

        super().__init__(n, r_min=r_min, r_max=r_max)

        n_ = self.n
        self.h = self.h[:n_]
        self.m = self.m[:n_]

    def transform(
        self,
        zj: np.ndarray,
        zk: np.ndarray,
        zl: np.ndarray,
        rijkl: np.ndarray,
        aijk: np.ndarray,
        ajkl: np.ndarray,
    ) -> np.ndarray:
        s4 = np.full(self.n, np.nan)

        cutoff_ijkl = cosine_cutoff(rijkl, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            s4[i] = np.sum(
                zj
                * zk
                * zl
                * (np.abs(np.cos(aijk)))
                * (np.abs(np.cos(ajkl)))
                * gaussian(rijkl, h, m)
                * cutoff_ijkl
            )
            i += 1

        return s4


class S5SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating five body (S5) terms.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
    ):
        """
        Args:
            n (int): The number of five body (S5) terms for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
        """

        super().__init__(n, r_min=r_min, r_max=r_max)

        n_ = self.n
        self.h = self.h[:n_]
        self.m = self.m[:n_]

    def transform(
        self,
        zj: np.ndarray,
        zk: np.ndarray,
        zl: np.ndarray,
        zm: np.ndarray,
        rijklm: np.ndarray,
        aijk: np.ndarray,
        ajkl: np.ndarray,
        aklm: np.ndarray,
    ) -> np.ndarray:
        s5 = np.full(self.n, np.nan)

        cutoff_ijklm = cosine_cutoff(rijklm, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            s5[i] = np.sum(
                zj
                * zk
                * zl
                * zm
                * (np.abs(np.cos(aijk)))
                * (np.abs(np.cos(ajkl)))
                * (np.abs(np.cos(aklm)))
                * gaussian(rijklm, h, m)
                * cutoff_ijklm
            )
            i += 1

        return s5


def cosine_cutoff(r: np.ndarray, r_max: float) -> np.ndarray:
    # returns a cosine cutoff function defined over `r` with `r_max` defining
    # the cutoff radius; see Behler; J. Chem. Phys., 2011, 134, 074106
    # (DOI: 10.1063/1.3553717)

    return (np.cos((np.pi * r) / r_max) + 1.0) / 2.0


def gaussian(r: np.ndarray, h: float, m: float) -> np.ndarray:
    # returns a gaussian-like function defined over `r` with eta (`h`) defining
    # the width and mu (`h`) defining the centrepoint/peak position

    return np.exp(-1.0 * h * (r - m) ** 2)


def gaussian_angle(r: np.ndarray, h: float, m: float) -> np.ndarray:
    # returns a gaussian-like function defined over `theta` with eta (`h`) defining
    # the width and mu (`h`) defining the centrepoint/peak position

    return np.exp(-1.0 * h * (r - m) ** 2)
