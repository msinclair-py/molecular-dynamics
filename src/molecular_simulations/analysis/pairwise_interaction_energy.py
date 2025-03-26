from copy import deepcopy
import MDAnalysis as mda
import mdtraj as md
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
import parmed as pmd
from tqdm import tqdm
from typing import Tuple

class PairwiseInteractionEnergy:
    def __init__(self, topology: str, trajectory: str, sel1: str, 
                 sel2: str, stride: int=1, datafile: str='energies.npy'):
        self.topology = topology

        if 'prmtop' in topology:
            self.top = AmberPrmtopFile(topology)
            self.system = self.top.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                                nonbondedCutoff=2. * nanometers,
                                                constraints=HBonds)
            self.top = self.top.topology
            self.add_hbonds = True
        elif 'pdb' in topology:
            self.top = PDBFile(topology).topology
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            self.system = forcefield.createSystem(self.top,
                                                  soluteDielectric=1.,
                                                  solventDielectric=80.)
            self.add_hbonds = False
        else:
            raise ValueError('Need prmtop or pdb for topology!')

        self.trajectory = trajectory
        
        # do indexing in MDA for superior atom selection language and out of memory
        # operation compared to mdtraj
        u = mda.Universe(self.topology, self.trajectory)
        idx1 = u.select_atoms(sel1).atoms.ix
        idx2 = u.select_atoms(sel2).atoms.ix
        self.sels = [idx1, idx2, np.concatenate((idx1, idx2))]

        self.stride = stride

        self.file = datafile

    def run(self) -> None:
        self.initialize_systems()
        self.compute_energy()
        self.split_off_components()
        self.write_out_energies()

    def subset_traj(self, sub_ind: list[str]) -> Tuple[Topology, System]:
        topology = md.Topology.from_openmm(self.top)
        sub_top = topology.subset(sub_ind)
        new_top = sub_top.to_openmm()

        structure = pmd.openmm.load_topology(self.top, self.system)[sub_ind]

        if self.add_hbonds: 
            hbond_type = pmd.topologyobjects.BondType(k=400, req=1.)
            constrained_bond_type = structure.bond_types.append(hbond_type)
            structure.bond_types.claim()

            for bond in structure.bonds:
                if bond.type is None:
                    bond.type = hbond_type

        new_system = structure.createSystem(implicitSolvent=GBn2,
                                            soluteDielectric=1.,
                                            solventDielectric=80.)
        
        return new_top, new_system

    def build_simulation_object(self, top: Topology, sys: System) -> Simulation:
        # cannot set 1 integrator to multiple simulation objects and also
        # unable to deepcopy the integrator object due to inability to pickle
        # a swig python object. must explicitly set each time
        integrator = LangevinMiddleIntegrator(
            300*kelvin, 
            1/picosecond, 
            0.004*picoseconds
        )
        simulation = Simulation(top, sys, integrator)
        return simulation
    
    def initialize_systems(self) -> None:
        simulations = [self.build_simulation_object(*self.subset_traj(sel)) 
                       for sel in self.sels]
        simulations.append(self.unset_charges(self.sels[-1]))
        self.simulations = simulations

    def unset_charges(self, sel: str) -> Simulation:
        top, sys = self.subset_traj(sel)
        forces = { force.__class__.__name__: force for force in sys.getForces() }
        nbforce = forces['NonbondedForce']

        for index in range(nbforce.getNumParticles()):
            [charge, sigma, epsilon] = nbforce.getParticleParameters(index)
            nbforce.setParticleParameters(index, 0, sigma, epsilon)

        return self.build_simulation_object(top, sys)

    def compute_energy(self) -> None:
        full_traj = md.load(self.trajectory, top=self.topology)
        n_frames = full_traj.n_frames // self.stride
        self.energies = np.zeros((n_frames, 4))
        for i, sim in tqdm(enumerate(self.simulations), total=3, 
                           position=0, leave=False, desc='System'):
            for fr in tqdm(range(n_frames), total=n_frames, position=1,
                           leave=False, desc='Frame'):
                frame = fr * self.stride
                try:
                    sel = self.sels[i]
                except IndexError:
                    sel = self.sels[-1]

                coords = full_traj.xyz[frame, sel, :]
                self.energies[fr, i] = self.calc_energy(sim, coords)
    
    def calc_energy(self, 
                    simulation: Simulation, 
                    positions: np.ndarray) -> float:
        simulation.context.setPositions(positions)
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()._value

    def split_off_components(self) -> None:
        total = self.energies[:, 2] - np.sum(self.energies[:, :2], axis=1)
        self.lj = self.energies[:, 3]
        self.el = total - self.lj

    def write_out_energies(self) -> None:
        shape = (self.lj.shape[0], 1)
        np.save(self.file, np.concatenate((self.lj.reshape(shape), 
                                           self.el.reshape(shape)), 
                                          axis=1))
