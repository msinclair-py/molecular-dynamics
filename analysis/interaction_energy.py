from abc import ABC, abstractmethod
from openmm import *
from openmm.app import *
from openmm.unit import *
import mdtraj as md
import numpy as np
import parmed as pmd
import gc
from tqdm import tqdm
from typing import Union

class InteractionEnergy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def energy(self):
        pass

    @abstractmethod
    def get_selection(self):
        pass

class StaticInteractionEnergy:
    """
    Computes the linear interaction energy between specified chain and other simulation
    components. Can specify a range of residues in chain to limit calculation to. Works on
    a static model but can be adapted to run on dynamics data.

    Inputs:
        pdb (str): Path to input PDB file
        chain (str): Defaults to A. The chain for which to compute the energy between.
            Computes energy between this chain and all other components in PDB file.
        first_residue (int, None): If set, will restrict the calculation to residues
            beginning with resid `first_residue`.
        last_residue (int, None): If set, will restrict the calculation to residues
            ending with resid `last_residue`.
    """
    def __init__(self, pdb: str, chain: str='A', platform: str='CUDA',
                 first_residue: Union[int, None]=None, 
                 last_residue: Union[int, None]=None):
        self.pdb = pdb
        self.chain = chain
        self.platform = Platform.getPlatformByName(platform)
        self.first = first_residue
        self.last = last_residue
        
    def get_system(self) -> None:
        pdb = PDBFile(self.pdb)
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        system = forcefield.createSystem(pdb.topology,
                                         soluteDielectric=1.,
                                         solventDielectric=80.)

        self.positions = pdb.positions
        self.get_selection(pdb.topology)

        return system

    def compute(self, positions: Union[np.ndarray, None]=None) -> None:
        self.lj = None
        self.coulomb = None

        system = self.get_system()
        if positions is None:
            positions = self.positions
            
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                force.setForceGroup(0)
                force.addGlobalParameter("solute_coulomb_scale", 1)
                force.addGlobalParameter("solute_lj_scale", 1)
                force.addGlobalParameter("solvent_coulomb_scale", 1)
                force.addGlobalParameter("solvent_lj_scale", 1)

                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    force.setParticleParameters(i, 0, 0, 0)
                    if i in self.selection:
                        force.addParticleParameterOffset("solute_coulomb_scale", i, charge, 0, 0)
                        force.addParticleParameterOffset("solute_lj_scale", i, 0, sigma, epsilon)
                    else:
                        force.addParticleParameterOffset("solvent_coulomb_scale", i, charge, 0, 0)
                        force.addParticleParameterOffset("solvent_lj_scale", i, 0, sigma, epsilon)

                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    force.setExceptionParameters(i, p1, p2, 0, 0, 0)

            else:
                force.setForceGroup(2)
        
        integrator = VerletIntegrator(0.001*picosecond)

        context = Context(system, integrator, self.platform)
        context.setPositions(positions)
        
        total_coulomb = self.energy(context, 1, 0, 1, 0)
        solute_coulomb = self.energy(context, 1, 0, 0, 0)
        solvent_coulomb = self.energy(context, 0, 0, 1, 0)
        total_lj = self.energy(context, 0, 1, 0, 1)
        solute_lj = self.energy(context, 0, 1, 0, 0)
        solvent_lj = self.energy(context, 0, 0, 0, 1)
        
        coul_final = total_coulomb - solute_coulomb - solvent_coulomb
        lj_final = total_lj - solute_lj - solvent_lj

        self.coulomb = coul_final.value_in_unit(kilocalories_per_mole)
        self.lj = lj_final.value_in_unit(kilocalories_per_mole)
    
    @staticmethod
    def energy(context, solute_coulomb_scale: int=0, solute_lj_scale: int=0, 
               solvent_coulomb_scale: int=0, 
               solvent_lj_scale: int=0) -> float:
        context.setParameter("solute_coulomb_scale", solute_coulomb_scale)
        context.setParameter("solute_lj_scale", solute_lj_scale)
        context.setParameter("solvent_coulomb_scale", solvent_coulomb_scale)
        context.setParameter("solvent_lj_scale", solvent_lj_scale)
        return context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
    
    def get_selection(self, topology) -> None:
        if self.first is None and self.last is None:
            selection = [a.index 
                        for a in topology.atoms() 
                        if a.residue.chain.id == self.chain]
        elif self.first is not None and self.last is None:
            selection = [a.index
                        for a in topology.atoms()
                        if a.residue.chain.id == self.chain 
                        and self.first <= a.residue.id]
        elif self.first is None:
            selection = [a.index
                        for a in topology.atoms()
                        if a.residue.chain.id == self.chain 
                        and self.last >= a.residue.id]
        else:
            selection = [a.index
                        for a in topology.atoms()
                        if a.residue.chain.id == self.sel_chain 
                        and self.first <= a.residue.id <= self.last]

        self.selection = selection

class InteractionEnergyFrame(StaticInteractionEnergy):
    def __init__(self, system: System, top: Topology, 
                 chain: str='A', platform: str='CUDA',
                 first_residue: Union[int, None]=None, 
                 last_residue: Union[int, None]=None):
        super().__init__('', chain, platform, first_residue, last_residue)
        self.system = system
        self.top = top

    def get_system(self):
        self.get_selection(self.top)
        return self.system


class DynamicInteractionEnergy:
    def __init__(self, top: str, traj: str, stride: int=1, 
                 chain: str='A', platform: str='CUDA',
                 first_residue: Union[int, None]=None,
                 last_residue: Union[int, None]=None,
                 progress_bar: bool=False):
        self.system = self.build_system(top, traj)
        self.coordinates = self.load_traj(top, traj)
        self.stride = stride
        self.progress = progress_bar

        self.IE = InteractionEnergyFrame(self.system, self.top, chain, 
                                         platform, first_residue, last_residue)

    def compute_energies(self) -> None:
        n_frames = self.coordinates.shape[0] // self.stride
        self.energies = np.zeros((n_frames, 2))
        
        if self.progress:
            pbar = tqdm(total=n_frames, position=0, leave=False)

        for i in range(n_frames):
            fr = i * self.stride
            self.IE.compute(self.coordinates[fr, :, :])
            self.energies[i, 0] = self.IE.lj
            self.energies[i, 1] = self.IE.coulomb

            if self.progress:
                pbar.update(1)

        if self.progress:
            pbar.close()
    
    def build_system(self, top: str, traj: str) -> System:
        if top[-3:] == 'pdb':
            top = PDBFile(top).topology
            self.top = top
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            return forcefield.createSystem(top, 
                                           soluteDielectric=1., 
                                           solventDielectric=80.)
        elif top[-6:] == 'prmtop':
            top = AmberPrmtopFile(top)
            self.top = top
            return top.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                    nonbondedCutoff=2. * nanometers,
                                    constraints=HBonds)
        else:
            raise NotImplementedError(f'Error! Topology type {top} not implemented!')

    def load_traj(self, top: str, traj: str) -> np.ndarray:
        return md.load(traj, top=top).xyz

    def setup_pbar(self) -> None:
        self.pbar = tqdm(total=self.coordinates.shape[0], position=0, leave=False)

class DynamicPotentialEnergy:
    """
    Class to compute the interaction energy from MD simulation using OpenMM.
    Inspired by: https://github.com/openmm/openmm/issues/3425
    """
    def __init__(self, top: str, traj: str, seltext: str='protein'):
        self.top = top
        self.traj = traj
        self.selection = seltext

        if top[-3:] == 'pdb':
            self.build_fake_topology(prmtop=False)
        elif top[-6:] == 'prmtop':
            self.build_fake_topology()
        else:
            raise NotImplementedError(f'Error! Topology type {top} not implemented!')
    
    def build_fake_topology(self, prmtop: bool=True) -> None:
        if prmtop:
            top = AmberPrmtopFile(self.top)
            system = top.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                      nonbondedCutoff=2. * nanometers,
                                      constraints=HBonds)
        else:
            top = PDBFile(self.top).topology
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            system = forcefield.createSystem(top,
                                             soluteDielectric=1.,
                                             solventDielectric=80.)
        
        # Load topology and subset
        topology = md.Topology.from_openmm(top)
        self.sub_ind = topology.select(self.selection)
        
        sub_top = topology.subset(self.sub_ind)
        self.old_topology = topology
        self.topology = sub_top.to_openmm()
        
        # Create protein only system
        structure = pmd.openmm.load_topology(top, system)
        sturcture = structure[self.sub_ind]

        # Add HBond restraints if in explicit water?
        if prmtop:
            new_bond_type = pmd.topologyobjects.BondType(k=400, req=1.)
            constrained_bond_type = structure.bond_types.append(new_bond_type)
            structure.bond_types.claim()

            for bond in structure.bonds:
                if bond.type is None:
                    bond.type = new_bond_type

            # Create new system
            new_system = structure.createSystem(nonbondedMethod=CutoffNonPeriodic, 
                                                nonbondedCutoff=2. * nanometers)

        else:
            new_system = structure.createSystem(soluteDielectric=1.,
                                                solventDielectric=80.)

        self.system = new_system
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004 * picoseconds)
        self.simulation = Simulation(self.topology, self.system, integrator)
    
    def compute(self) -> None:
        full_traj = md.load(self.traj, top=self.top)
        self.energies = np.zeros((full_traj.n_frames))
        for fr in range(full_traj.n_frames):
            energy = self.calc_energy(full_traj.xyz[fr, self.sub_ind, :])
            self.energies[fr] = energy._value

    def calc_energy(self, positions) -> float:
        self.simulation.context.setPositions(positions)
        state = self.simulation.context.getState(getEnergy=True)

        return state.getPotentialEnergy()

class PairwiseInteractionEnergy(StaticInteractionEnergy):
    """
    Computes the pairwise interaction energy between a single residue from one 
    selection and the entirety of another selection.
    """
    def __init__(self, system: System, top: Topology, 
                 mda_sel1: str, mda_sel2: str,
                 chain: str='A', platform: str='CUDA'):
        super().__init__('', chain, platform, None, None)
        self.system = system
        self.top = top

    def get_system(self):
        self.get_selection(self.top)
        return self.system

    def get_selection(self):
        self.selection = pass
