import numpy as np
from openmm import unit as u
from pathlib import Path

class RCReporter:
    """Custom reaction-coordinate reporter for OpenMM. Computes reaction
    coordinate progress for a given frame, and reports the target, rc0,
    current state, rc, and both distances that comprise the reaction 
    coordinate, d_ik, d_ij.
    """
    def __init__(self,
                 file: Path,
                 report_interval: int,
                 atom_indices: list[int],
                 rc0: float):
        self.file = open(file, 'w')
        self.file.write('rc0,rc,dist_ik, dist_jk\n')
        
        self.report_interval = report_interval
        self.atom_indices = atom_indices
        self.rc0 = rc0
        
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
        box_vecs = state.getPeriodicBoxVectors(asNumpy=True)
        pos = state.getPositions(asNumpy=True)
        
        i, j, k = self.atom_indices
        dist_ik = np.linalg.norm(pos[i] - pos[k])
        dist_jk = np.linalg.norm(pos[j] - pos[k])
        
        rc = dist_ik - dist_jk

        self.file.write(f'{self.rc0},{rc},{dist_ik},{dist_jk}\n')
        self.file.flush()
