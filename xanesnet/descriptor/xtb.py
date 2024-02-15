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

from ase import Atoms
from tblite.interface import Calculator
from xanesnet.descriptor.base_descriptor import BaseDescriptor


###############################################################################
################################## CLASSES ####################################
###############################################################################


class XTB(BaseDescriptor):
    def __init__(
        self,
        method: str = "GFN2-xTB",
        accuracy: float = 1.0,
        guess: int = 0,
        max_iter: int = 250,
        mixer_damping: float = 0.4,
        save_integrals: int = 0,
        temperature: float = 9.5e-4,
        verbosity: int = 1,
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
            verbosity (int): Set verbosity of printout
                Defaults to 1
        """
        self.method = method
        self.accuracy = accuracy
        self.guess = guess
        self.max_iter = max_iter
        self.mixer_damping = mixer_damping
        self.save_integrals = save_integrals
        self.temperature = temperature
        self.verbosity = verbosity

    def transform(self, system: Atoms):
        numbers = system.get_atomic_numbers()
        positions = system.get_positions()
        # Convert atomic unit to bohr
        positions = positions * 1.8897259886

        calc = Calculator(self.method, numbers, positions)

        calc.set("save-integrals", self.save_integrals)
        calc.set("accuracy", self.accuracy)
        calc.set("guess", self.guess)
        calc.set("max-iter", self.max_iter)
        calc.set("mixer-damping", self.mixer_damping)
        calc.set("temperature", self.temperature)
        calc.set("verbosity", self.verbosity)

        # calc = Calculator(
        #     method="GFN2-xTB",
        #     numbers=np.array([14, 1, 1, 1, 1]),
        #     positions=np.array(
        #         [
        #             [0.000000000000, 0.000000000000, 0.000000000000],
        #             [1.617683897558, 1.617683897558, -1.617683897558],
        #             [-1.617683897558, -1.617683897558, -1.617683897558],
        #             [1.617683897558, -1.617683897558, 1.617683897558],
        #             [-1.617683897558, 1.617683897558, 1.617683897558],
        #         ]
        #     ),
        # )
        try:
            res = calc.singlepoint()
            res.get("energy")  # Results in atomic units
        except RuntimeError as e:
            print(f"RuntimeError: {e}")

    def get_number_of_features(self):
        return 0
