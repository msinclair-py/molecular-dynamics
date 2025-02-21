from copy import deepcopy
import MDAnalysis as mda
import mdtraj as md
from openmm import *
from openmm.app import *
from openmm.unit import *

class PairwiseInteractionEnergy:
    def __init__(self, topology: str, trajectory: str, sel1: str, sel2: str):
        if 'prmtop' in topology:
            self.top = AmberPrmtopFile(topology)
            self.system = self.top.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                                nonbondedCutoff=2. * nanometers,
                                                constraints=HBonds)
        elif 'pdb' in topology:
            self.top = PDBFile(topology)
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            self.system = forcefield.createSystem(self.top,
                                                  soluteDielectric=1.,
                                                  solventDielectric=80.)
        else:
            raise ValueError('Need prmtop or pdb for topology!')

        self.traj = trajectory
        self.integrator = LangevinMiddleIntegrator(
            300*kelvin, 
            1/picosecond, 
            0.004*picoseconds
        )
        
        idx1 = ()
        idx2 = ()
        self.sels = [idx1, idx2, idx1 + idx2]

    def subset_traj(self, sub_ind: list[str]):
        topology = mm.Topology.from_openmm(self.top)
        sub_top = topology.subset(sub_ind)
        new_top = sub_top.to_openmm()

        structure = pmd.openmm.load_topology(self.top, self.system)[sub_ind]
        new_system = structure.createSystem(soluteDielectric=1.,
                                            solventDielectric=80.)
        simulation = Simulation(new_top, new_system, self.integrator)
        return simulation

    def initialize_systems(self):
        simulations = [self.subset_traj(sel) for sel in self.sels]
        simulations.append(deepcopy(simulations[-1]))
        self.simulations = simulations

    def compute_energy(self):
        full_traj = md.load(self.traj, top=self.top)
        self.energies = np.zeros((full_traj.n_frames, 4))
        for i, sim in enumerate(self.simulations):
            for fr in range(full_traj.n_frames):
                try:
                    sel = self.sels[i]
                except IndexError:
                    sel = self.sels[-1]

                coords = full_traj.xyz[fr, sel, :]
                self.energies[fr, i] = self.calc_energy(sim, coords)
    
    def calc_energy(self, 
                    simulation: Simulation, 
                    positions: np.ndarray) -> float:
        simulation.context.setPositions(positions)
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()._value

    def split_off_components(self):
        total = self.energies[:, 3] - np.sum(self.energies[:, :2], axis=0)
        self.lj = self.energies[:, 4]
        self.el = total - self.lj
