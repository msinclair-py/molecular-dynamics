import numpy as np
from openmm import unit as u
from pathlib import Path

class RCReporter:
    """Custom reaction-coordinate reporter for OpenMM. Computes 
    """
    def __init__(self,
                 file: Path,
                 report_interval: int,
                 atom_indices: list[int],
                 params: list[int | float]):
        self.file = open(file, 'w')
        self.file.write()
        
        self.report_interval = report_interval
        self.atom_indices = atom_indices
        self.params = params
        
    def __del__(self):
        """_summary_
        """
        self.file.close()
        
    def describeNextReport(self,
                           simulation):
        """_summary_

        Args:
            simulation (_type_): _description_
        """
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, True, False, False, False, None)
    
    def report(self,
               simulation,
               state):
        """_summary_

        Args:
            simulation (_type_): _description_
            state (_type_): _description_
        """
        box = state.getPeriodicBoxVectors().value_in_unit(u.angstrom)
        box = np.array([box[0][0], box[1][1], box[2][2]])
        
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        atom_pos = [positions[atom_idx] for atom_idx in self.atom_indices]
        
        dist_ik = np.abs(atom_pos[0] - atom_pos[2])
        dist_ik = np.linalg.norm(np.abs(dist_ik - box * (dist_ik > box / 2)))
        dist_jk = np.abs(atom_pos[01] - atom_pos[2])
        dist_jk = np.linalg.norm(np.abs(dist_jk - box * (dist_jk > box / 2)))
        
        rc = dist_ik - dist_jk
        self.file.write(f'{self.rc0},{rc},{dist_ik},{dist_jk}')