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

from .vector_descriptor import VectorDescriptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class ARMSR(VectorDescriptor):
    """
    A class for transforming a molecular system into a multiple scattering
    descriptor. AR-MSRs encode the local geometry
    around an absorption site in a manner reminescent of the path expansion in
    multiple scattering theory. Here, in contrast to the MSR vector, we have an 
    angular grid as well as radial grid so information is not overly compressed
    In contrast to MSR, we have truncated the expansion to S3, so that the vector
    does not explode in legnth, but these higher-order terms can easily be 
    included. 

    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 8.0,
        n_s2: int = 0,
        n_s3: int = 0,
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

        rij = system.get_distances(ij[:, 0], ij[:, 1])

        armsr = [] 

        if self.n_s2:
            zj = system.get_atomic_numbers()[ij[:, 1]]
            zj = 0.1*zj
            rij = system.get_distances(ij[:, 0], ij[:, 1])
            s2 = self.s2_transformer.transform(zj, rij)
            armsr = np.append(armsr, s2)

        if self.n_s3:
            zj = system.get_atomic_numbers()[ijk[:, 1]]
            zk = system.get_atomic_numbers()[ijk[:, 2]]
            zj = 0.1*zj 
            zk = 0.1*zk 
            rij = system.get_distances(ijk[:, 0], ijk[:, 1])
            rik = system.get_distances(ijk[:, 1], ijk[:, 2])
            rjk = system.get_distances(ijk[:, 0], ijk[:, 2])
            rijk = rij + rik
            aijk = np.radians(system.get_angles(ijk))
            s3 = self.s3_transformer.transform(zj, zk, rijk, aijk)
            armsr = np.append(armsr, s3)

        if self.use_spin:
            armsr = np.append(system.info["S"], armsr)

        if self.use_charge:
            armsr = np.append(system.info["q"], armsr)

        return armsr 

    def get_number_of_features(self) -> int:
        return int(self.n_s2 + (self.n_s3*18) + self.use_charge + self.use_spin)

    def process(self, atoms: Atoms):
        return self.transform(atoms)

class SymmetryFunctionTransformer(ABC):
    """
    An abstract base class for generating angle resolved multiple scattering vector.
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
    A class for generating three body (S3) terms for encoding. 
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

        n = n * 18
        self.n = self.n * 18
        n_ = self.n
        self.h = self.h[:n_]
        self.m = self.m[:n_]
        self.th = self.th[:n_]
        self.tm = self.tm[:n_]

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
            for th, tm in zip(self.th, self.tm):
                s3[i] = np.sum(
                    zj
                    * zk
                    * gaussian_angle(np.cos(aijk), th, tm)
                    * gaussian(rijk, h, m)
                    * cutoff_ijk
                ) 
                i += 1

        return s3

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
