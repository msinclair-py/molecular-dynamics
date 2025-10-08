from .omm_simulator import Simulator
from openmm import *
from openmm.app import *

class EVB:
    def __init__(self,):
        self.sim_engine = Simulator()
    
    @staticmethod
    def umbrella_force(atom_i: int, 
                       atom_j: int, 
                       atom_k: int,
                       k: float, 
                       rc0: float) -> CustomBondForce:
        """Difference of distances umbrella force. Think pulling an oxygen off

        Args:
            atom_i (int): Index of first atom participating (from reactant).
            atom_j (int): Index of second atom participating (from product).
            atom_k (int): Index of shared atom participating in both reactant and product.
            k (float, optional): Harmonic spring constant.
            rc0 (float, optional): Target equilibrium distance for current window.

        Returns:
            CustomBondForce: Force that drives sampling in each umbrella window.
        """
        force = CustomCompoundBondForce('0.5 * k * ((r13 - r23) - rc0) ^ 2; r13=distance(p1, p3); r23=distance(p2, p3);')
        force.addGlobalParameter('k', k)
        force.addGlobalParameter('rc0', rc0)
        force.addBond([atom_i, atom_j, atom_k])
    
        return force
    
    @staticmethod
    def morse_bond_force(atom_i: int,
                         atom_j: int,
                         D_e: float,
                         alpha: float,
                         r0: float) -> CustomBondForce:
        """Generates a custom Morse potential between two atom indices.

        Args:
            atom_i (int): Index of first atom
            atom_j (int): Index of second atom
            D_e (float, optional): Depth of the Morse potential.
            alpha (float, optional): Stiffness of the Morse potential.
            r0 (float, optional): Equilibrium distance of the bond represented.

        Returns:
            CustomBondForce: Force corresponding to a Morse potential.
        """
        force = CustomBondForce('D_e * (1 - exp(-alpha * (r-r0))) ^ 2')
        force.addGlobalParameter('D_e', D_e)
        force.addGlobalParameter('alpha', alpha)
        force.addGlobalParameter('r0', r0)
        force.addBond(atom_i, atom_j)
        
        return force