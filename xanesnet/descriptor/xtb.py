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

from tblite.interface import Calculator
from ase import Atoms
import numpy as np

from xanesnet.descriptor import WACSF

###############################################################################
################################## CLASSES ####################################
###############################################################################


class XTB(WACSF):
    """
    A class for transforming a molecular system into a project density of
    states representation from XTB model.
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 6.0,
        method: str = "GFN2-xTB",
        accuracy: float = 1.0,
        guess: int = 0,
        max_iter: int = 250,
        mixer_damping: float = 0.4,
        save_integrals: int = 0,
        temperature: float = 9.5e-4,
        verbosity: int = 0,
        e_min: float = -20.0,
        e_max: float = 20.0,
        sigma: float = 0.7,
        num_points: float = 200,
        use_wacsf=False,
        n_g2: int = 0,
        n_g4: int = 0,
        l: list = [1.0, -1.0],
        z: list = [1.0],
        g2_parameterisation: str = "shifted",
        g4_parameterisation: str = "centred",
        use_charge=False,
        use_spin=False,
        use_quad=False,
        use_occupied=False,
    ):
        """
        Args:
            accuracy (float): Numerical thresholds for SCC.
                Defaults to 1.0.
            guess (int): Initial guess for wavefunction.
                Defaults to 0 (SAD).
            max_iter (int): Maximum number of SCC iterations.
                Defaults to 250.
            mixer_damping (float): Parameter for the SCC mixer.
                Defaults to 0.4.
            save_integrals (int): Keep integral matrices in results.
                Defaults to 0 (False).
            temperature (float): Electronic temperature for filling.
                Defaults to 9.500e-4.
            verbosity (float): Set verbosity of printout
                Defaults to 0
        """
        if use_wacsf:
            super().__init__(
                r_min,
                r_max,
                n_g2,
                n_g4,
                l,
                z,
                g2_parameterisation,
                g4_parameterisation,
                use_charge,
                use_spin,
            )

        self.use_wacsf = use_wacsf
        self.use_spin = use_spin
        self.use_charge = use_charge
        self.use_quad = use_quad
        self.use_occupied = use_occupied
        self.method = method
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.e_min = e_min
        self.e_max = e_max
        self.num_points = num_points
        self.sigma = sigma

    def transform(self, system: Atoms):
        numbers = system.get_atomic_numbers()
        nelectron = np.sum(numbers)
        positions = system.get_positions()
        positions = positions * 1.8897259886

        if (self.use_spin and not self.use_charge) or (not self.use_spin and self.use_charge):
            err_str = (
                "For the p-DOS descriptor, it is not a good idea to only"
                "consider overall charge or spin state. Both should be"
                "include simultaneously or not at all."
            )
            raise NotImplementedError(err_str)

        else:

            if self.use_spin and self.use_charge:
                charge = system.info["q"]
                spin = system.info["s"]
                if (((nelectron - charge) % 2) == 1) and (spin  % 2) == 0:
                   err_str = (
                       "The number of electrons is inconsistent with the spin"
                       "state you have defined."
                   )
                   raise ValueError(err_str)
                elif (((nelectron - charge) % 2) == 0) and (spin  % 2) == 1:
                   err_str = (
                       "The number of electrons is inconsistent with the spin"
                       "state you have defined."
                   )
                   raise ValueError(err_str)
            else:
                charge  = 0
                spin    = 0

        calc = Calculator(self.method, numbers, positions, charge, spin)
        calc.set("verbosity", self.verbosity)
        calc.set("max-iter", self.max_iter)

        res = calc.singlepoint()
        res.get("energy") 
        coeff = res.get("orbital-coefficients")
        coeff = np.square(coeff)
        if (numbers[0] >= 21) and (numbers[0] <= 29): 
            p_dos = np.array([np.sum(coeff[6:8, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
        elif (numbers[0] >= 39) and (numbers[0] <= 47):
            p_dos = np.array([np.sum(coeff[6:8, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
        elif (numbers[0] >= 57) and (numbers[0] <= 79):
            p_dos = np.array([np.sum(coeff[6:8, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
        elif (numbers[0] >= 89) and (numbers[0] <= 112):
            p_dos = np.array([np.sum(coeff[6:8, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
        else:
            p_dos = np.array([np.sum(coeff[1:3, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
        
        orbe = np.multiply(res.get("orbital-energies"), 27.211324570273)
        orbo = res.get("orbital-occupations")
        if self.use_occupied:
            unoccupied_pdos = p_dos * np.abs(orbo)
        else:
            unoccupied_pdos = p_dos * np.abs((orbo - 2))
        unoccupied_orbital_energies = orbe

        # Generate a grid and broaden pDOS
        x = np.linspace(self.e_min, self.e_max, num=self.num_points, endpoint=True)
        sigma = self.sigma
        pdos_gauss = spectrum(unoccupied_orbital_energies, unoccupied_pdos, sigma, x)
        pdos_gauss = np.multiply(pdos_gauss,10)

        if self.use_quad:
            if (numbers[0] >= 21) and (numbers[0] <= 29):
                d_dos = np.array([np.sum(coeff[0:4, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
            elif (numbers[0] >= 39) and (numbers[0] <= 47):
                d_dos = np.array([np.sum(coeff[0:4, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
            elif (numbers[0] >= 57) and (numbers[0] <= 79):
                p_dos = np.array([np.sum(coeff[0:4, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
            elif (numbers[0] >= 89) and (numbers[0] <= 112):
                d_dos = np.array([np.sum(coeff[0:4, i]) / np.sum(coeff[:, i]) for i in range(len(coeff))])
            else:
                err_str = ("d-orbitals are not considered for these atoms.")
                raise ValueError(err_str)
    
            orbe = np.multiply(res.get("orbital-energies"), 27.211324570273)
            orbo = res.get("orbital-occupations")
            if self.use_occupied:
                unoccupied_pdos = p_dos * np.abs(orbo)
            else:
                unoccupied_pdos = p_dos * np.abs((orbo - 2))
            unoccupied_orbital_energies = orbe
    
            # Generate a grid and broaden dDOS
            x = np.linspace(self.e_min, self.e_max, num=self.num_points, endpoint=True)
            sigma = self.sigma
            ddos_gauss = spectrum(unoccupied_orbital_energies, unoccupied_ddos, sigma, x)
            ddos_gauss = np.multiply(ddos_gauss,10)

        pdos_gauss = np.append(pdos_gauss,ddos_gauss)

        if self.use_wacsf:
            pdos_gauss = np.append(pdos_gauss, super().transform(system))

        return pdos_gauss

    def get_number_of_features(self):
        if self.use_wacsf:
            if self.use_quad:
                return int(
                    self.num_points
                    + self.num_points
                    + 1
                    + self.n_g2
                    + self.n_g4
                )
            else:
                return int(
                    self.num_points
                    + 1
                    + self.n_g2
                    + self.n_g4
                )
        else:
            if self.use_quad:
               return int(self.num_points + self.num_points)
            else:
               return int(self.num_points)

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
