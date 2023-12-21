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
from pyscf import scf,gto

from ase import Atoms
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

from xanesnet.descriptor.vector_descriptor import VectorDescriptor

###############################################################################
################################## CLASSES ####################################
###############################################################################


class PDOS(VectorDescriptor):
    """
    A class for transforming a molecular system into a project density of
    states representation.
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 6.0,
        e_min: float = -20.0,
        e_max: float = 20.0,
        sigma: float = 0.7,
        orb_type: str = "p",
        num_points: float =  200,
        basis: str = "3-21g",
        init_guess: str = "minao",
        max_scf_cycles: float = 0,
        use_wacsf=False,
        n_g2: int = 0,
        n_g4: int = 0,
        l: list = [1.0, -1.0],
        z: list = [1.0],
        g2_parameterisation: str = "shifted",
        g4_parameterisation: str = "centred",
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
            n_g2 (int, optional): The number of G2 symmetry functions to use
                for encoding.
                Defaults to 0.
            n_g4 (int, optional): The number of G4 symmetry functions to use
                for encoding.
                Defaults to 0.
            l (list, optional): List of lambda values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0, -1.0].
            z (list, optional): List of zeta values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0].
            g2_parameterisation (str, optional): The strategy to use for G2
                symmetry function parameterisation; choices are 'shifted' or
                'centred'. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
            g4_parameterisation (str, optional): The strategy to use for G4
                symmetry function parameterisation; choices are 'shifted' or
                'centred'. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to 'centred'.
            e_min (float, optional): The minimum energy grid point for the pDOS (in eV)
                Default: -20.0 eV.
            e_max (float, optional): The maximum energy grid point for the pDOS (in eV)
                Default: 20.0 eV.
            sigma (float, optional): This is the FWHM of the Gaussian function used to  
                broaden the pDOS obtained from pySCF.
                Default: 0.7 eV.
            num_points (float, optional): This is the number of point over which the broadened
                pDOS is projected. 
                Default: 200.
            basis (string, optional): This is the basis set used by pySCF during developing
                the pDOS.
                Default: 3-21G
            basis (string, optional): Defines the method of the initial guess used by pySCF 
                during generation of the pDOS.
                Default: minao 
            max_scf_cycles (float, optional): This is the number of SCF cycles used by pySCF 
                during develop the pDOS. Smaller numbers will be closer to the raw guess, while
                larger number will take longer to load.
                Note, the warnings are suppressed and so it will not tell you if the SCF is 
                converged. Larger numbers make this more likely, but do not gurantee it.
                Default: 0
            use_wacsf (bool): If True, the wacsf descriptor for the structure is also generated 
                and concatenated onto the end after the pDOS descriptor.
                Defaults to False.
            use_charge (bool): If True, includes an additional element in the
                vector descriptor for the charge state of the complex.
                Defaults to False.
            use_spin (bool): If True, includes an additional element in the
                vector descriptor for the spin state of the complex.
                Defaults to False.
        """

        super().__init__(r_min, r_max, use_charge, use_spin)

        self.e_min = e_min 
        self.e_max = e_max 
        self.num_points = num_points 
        self.max_scf_cycles = max_scf_cycles 
        self.basis = basis
        self.sigma = sigma
        self.init_guess = init_guess
        self.orb_type = orb_type
        self.use_wacsf = use_wacsf

        if self.use_wacsf: 
            self.n_g2 = n_g2
            self.n_g4 = n_g4
            if self.n_g4:
                self.l = l
                self.z = z
            self.g2_parameterisation = g2_parameterisation
            self.g4_parameterisation = g4_parameterisation
            
            if self.n_g2:
                self.g2_transformer = G2SymmetryFunctionTransformer(
                    self.n_g2,
                    r_min=self.r_min,
                    r_max=self.r_max,
                    parameterisation=self.g2_parameterisation,
                )
            
            if self.n_g4:
                self.g4_transformer = G4SymmetryFunctionTransformer(
                    self.n_g4,
                    r_min=self.r_min,
                    r_max=self.r_max,
                    l=self.l,
                    z=self.z,
                    parameterisation=self.g4_parameterisation,
                )

    def transform(self, system: Atoms) -> np.ndarray:

# Move molecular geometry from ASE environment to pySCF format
        mol = gto.Mole()
        mol.atom = atoms_to_pyscf(system)
        mol.basis = self.basis 
#       charge=0
#       spin=0
        mol.build()

# Create a SCF (Self-Consistent Field) object with a specific max_cycle value
        max_scf_cycles = self.max_scf_cycles 
        mf = scf.RHF(mol)
        mf.init_guess = self.init_guess
        mf.max_cycle = max_scf_cycles 

# Perform the SCF calculation, suppress warnings as we know SCF isn't converged!
        mf.verbose = 0
        mf.kernel()

# Get the atomic orbital coefficients for molecular orbitals
        ao_coefficients = mf.mo_coeff
        ao_labels = mol.ao_labels()

# Get the orbital energies
        orbital_energies = mf.mo_energy
        pdos = np.zeros_like(orbital_energies)

# Calculate the squared magnitude of coefficients and convert to percentages
        coeff_magnitude_squared = np.square(ao_coefficients)
        coeff_percentage = (coeff_magnitude_squared / np.sum(coeff_magnitude_squared, axis=0)) 

# Find the index of the first atom's (absorbing atom) atomic orbital labels
        first_atom_index = ao_labels.index([label for label in ao_labels if label.split()[0] == '1'][0])

# Find coeff_percent for absorbing atom
        for i, coeff_percent in enumerate(coeff_percentage.T):  # Transpose for easier iteration
             p_contribution = 0
             for j, percent in enumerate(coeff_percent):
                  if j < first_atom_index:
                      label = ao_labels[j]
                      atomic_num, orb_type = label.split()[1], label.split()[2]
                      if self.orb_type in orb_type:
                         p_contribution = p_contribution + percent
             pdos[i] = p_contribution

# Convert orbital energies from atomic units to eV
        orbital_energies = np.multiply(orbital_energies,27.211324570273)
# Filter out the occupied orbitals 
        unoccupied_orbital_energies = orbital_energies[mol.nelectron // 2:]
        unoccupied_pdos = pdos[mol.nelectron // 2:]

# Generate a grid and broaden pDOS 
        x = np.linspace(self.e_min,self.e_max,num=self.num_points, endpoint=True)
        sigma = self.sigma
        gE = spectrum(unoccupied_orbital_energies, unoccupied_pdos, sigma, x)

        pdos_gauss = gE
#       pdos_gauss = np.multiply(pdos_gauss,20)

        if self.use_wacsf:
            rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
            system = system[rij_in_range]
       
            ij = np.array([[0, j] for j in range(1, len(system))], dtype="uint16")
       
            if self.n_g4:
                jik = np.array(
                    [
                        [j, 0, k]
                        for j in range(1, len(system))
                        for k in range(1, len(system))
                        if k > j
                    ],
                    dtype="uint16",
                )
       
            rij = system.get_distances(ij[:, 0], ij[:, 1])
            g1 = np.sum(cosine_cutoff(rij, self.r_max))
       
#           wacsf = g1
            wacsf = []
       
            if self.n_g2:
                zj = system.get_atomic_numbers()[ij[:, 1]]
                zj = 0.1*zj
                rij = system.get_distances(ij[:, 0], ij[:, 1])
                g2 = self.g2_transformer.transform(zj, rij)
                wacsf = np.append(wacsf, g2)
       
            if self.n_g4:
                zj = system.get_atomic_numbers()[jik[:, 0]]
                zk = system.get_atomic_numbers()[jik[:, 2]]
                zj = 0.1*zj
#               zk = 0.1*zk
                rij = system.get_distances(jik[:, 1], jik[:, 0])
                rik = system.get_distances(jik[:, 1], jik[:, 2])
                rjk = system.get_distances(jik[:, 0], jik[:, 2])
                ajik = np.radians(system.get_angles(jik))
                g4 = self.g4_transformer.transform(zj, zk, rij, rik, rjk, ajik)
                wacsf = np.append(wacsf, g4)

            pdos_gauss = np.append(pdos_gauss,wacsf)
           
        if self.use_spin:
            pdos_gauss = np.append(system.info["S"], pdos_gauss)

        if self.use_charge:
            pdos_gauss = np.append(system.info["q"], pdos_gauss)

        return pdos_gauss

    def get_number_of_features(self) -> int:
        if self.use_wacsf: 
            return int(self.num_points + self.n_g2 + self.n_g4 + self.use_charge + self.use_spin)
        else:
            return int(self.num_points  + self.use_charge + self.use_spin )

    def process(self, atoms: Atoms):
        return self.transform(atoms)

class SymmetryFunctionTransformer(ABC):
    """
    An abstract base class for generating symmetry function vectors.
    """

    def __init__(self, n: int, r_min: float, r_max: float, parameterisation: str):
        """
        Args:
            n (int): The number of symmetry functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            parameterisation (str, optional): The strategy to use for symmetry
                function parameterisation; choices are 'shifted' or 'centred'.
                For details, see Marquetand et al.; J. Chem. Phys., 2018, 148,
                241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
        """

        self.n = n
        self.r_min = r_min
        self.r_max = r_max
        self.parameterisation = parameterisation

        if self.parameterisation == "shifted":
            r_aux = np.linspace(self.r_min + 0.5, self.r_max - 0.5, self.n)
            dr = np.diff(r_aux)[0]
            self.h = np.array([1.0 / (2.0 * (dr**2)) for _ in r_aux])
            self.m = np.array([i for i in r_aux])
        elif self.parameterisation == "centred":
            r_aux = np.linspace(self.r_min + 1.0, self.r_max - 0.5, self.n)
            self.h = np.array([1.0 / (2.0 * (i**2)) for i in r_aux])
            self.m = np.array([0.0 for _ in r_aux])
        else:
            err_str = (
                "parameterisation options: 'shifted' | 'centred'; "
                "for details, see DOI: 10.1063/1.5019667"
            )
            raise ValueError(err_str)

    @abstractmethod
    def transform(self, *args) -> np.ndarray:
        """
        Encodes structural information (`*args`) using symmetry functions.

        Returns:
            np.ndarray: A symmetry function vector.
        """

        pass


class G2SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating G2 symmetry function vectors.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
        parameterisation: str = "shifted",
    ):
        """
        Args:
            n (int): The number of G2 symmetry functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            parameterisation (str, optional): The strategy to use for symmetry
                function parameterisation; choices are 'shifted' or 'centred'.
                For details, see Marquetand et al.; J. Chem. Phys., 2018, 148,
                241709 (DOI: 10.1063/1.5019667).
                Defaults to 'shifted'.
        """

        super().__init__(n, r_min=r_min, r_max=r_max, parameterisation=parameterisation)

    def transform(self, zj: np.ndarray, rij: np.ndarray) -> np.ndarray:
        g2 = np.full(self.n, np.nan)

        cutoff_ij = cosine_cutoff(rij, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            g2[i] = np.sum(zj * gaussian(rij, h, m) * cutoff_ij)
            i += 1

        return g2


class G4SymmetryFunctionTransformer(SymmetryFunctionTransformer):
    """
    A class for generating G4 symmetry function vectors.
    """

    def __init__(
        self,
        n: int,
        r_min: float = 0.0,
        r_max: float = 8.0,
        l: list = [1.0, -1.0],
        z: list = [1.0],
        parameterisation: str = "centred",
    ):
        """
        Args:
            n (int): The number of G4 symmetry functions to use for encoding.
            r_min (float, optional): The minimum radial cutoff distance (in A)
                around the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float, optional): The maximum radial cutoff distance (in A)
                around the absorption site.
                Defaults to 8.0.
            l (list, optional): List of lambda values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0, -1.0].
            z (list, optional): List of zeta values for G4 symmetry function
                encoding. For details, see Marquetand et al.; J. Chem. Phys.,
                2018, 148, 241709 (DOI: 10.1063/1.5019667).
                Defaults to [1.0].
            parameterisation (str, optional): The strategy to use for symmetry
                function parameterisation; choices are 'shifted' or 'centred'.
                For details, see Marquetand et al.; J. Chem. Phys., 2018, 148,
                241709 (DOI: 10.1063/1.5019667).
                Defaults to 'centred'.
        """

        super().__init__(n, r_min=r_min, r_max=r_max, parameterisation=parameterisation)

        if self.n % (len(l) * len(z)):
            err_str = (
                f"can't generate {self.n} G4 symmetry functions with "
                f"{len(l)} lambda and {len(z)} zeta value(s)"
            )
            raise ValueError(err_str)
        else:
            n_ = int(self.n / (len(l) * len(z)))
            self.h = self.h[:n_]
            self.m = self.m[:n_]
            self.l = np.array(l)
            self.z = np.array(z)

    def transform(
        self,
        zj: np.ndarray,
        zk: np.ndarray,
        rij: np.ndarray,
        rik: np.ndarray,
        rjk: np.ndarray,
        ajik: np.ndarray,
    ) -> np.ndarray:
        g4 = np.full(self.n, np.nan)

        cutoff_ij = cosine_cutoff(rij, self.r_max)
        cutoff_ik = cosine_cutoff(rik, self.r_max)
        cutoff_jk = cosine_cutoff(rjk, self.r_max)

        i = 0
        for h, m in zip(self.h, self.m):
            for l in self.l:
                for z in self.z:
                    g4[i] = np.sum(
                        zj
                        * zk
                        * (1.0 + (l * np.cos(ajik))) ** z
                        * gaussian(rij, h, m)
                        * cutoff_ij
                        * gaussian(rik, h, m)
                        * cutoff_ik
                        * gaussian(rjk, h, m)
                        * cutoff_jk
                    ) * (2.0 ** (1.0 - z))
                    i += 1

        return g4

def cosine_cutoff(r: np.ndarray, r_max: float) -> np.ndarray:
    # returns a cosine cutoff function defined over `r` with `r_max` defining
    # the cutoff radius; see Behler; J. Chem. Phys., 2011, 134, 074106
    # (DOI: 10.1063/1.3553717)

    return (np.cos((np.pi * r) / r_max) + 1.0) / 2.0


def gaussian(r: np.ndarray, h: float, m: float) -> np.ndarray:
    # returns a gaussian-like function defined over `r` with eta (`h`) defining
    # the width and mu (`h`) defining the centrepoint/peak position

    return np.exp(-1.0 * h * (r - m) ** 2)

def atoms_to_pyscf(atoms):
    # This converts system in ASE format into atom format for pySCF
    atom_list = []
    for atom in atoms:
        symbol = atom.symbol
        position = atom.position
        atom_list.append((symbol, tuple(position)))
    return atom_list

def spectrum(E, osc, sigma, x):
    # This Gaussian broadens the partial density of states over a defined 
    # energy range and grid spacing. 
    gE = []
    for Ei in x:
        tot = 0
        for Ej, os in zip(E, osc):
            tot += os * np.exp(-((((Ej - Ei) / sigma) ** 2)))
        gE.append(tot)
    return gE
