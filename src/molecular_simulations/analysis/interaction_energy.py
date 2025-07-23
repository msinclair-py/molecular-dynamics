from abc import ABC, abstractmethod
from copy import deepcopy
from openmm import *
from openmm.app import *
from openmm.unit import *
import MDAnalysis as mda
from MDAnalysis.analysis.distances import contact_matrix
import mdtraj as md
import numpy as np
import parmed as pmd
from pathlib import Path
from pdbfixer import PDBFixer
import pickle
import gc
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

PathLike = Union[Path, str]

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

class StaticInteractionEnergy(InteractionEnergy):
    """
    Computes the linear interaction energy between specified chain and other simulation
    components. Can specify a range of residues in chain to limit calculation to. Works on
    a static model but can be adapted to run on dynamics data.

    Inputs:
        pdb (str): Path to input PDB file
        chain (str): Defaults to A. The chain for which to compute the energy between.
            Computes energy between this chain and all other components in PDB file.
            Set to a whitespace if there are no chains in your PDB.
        platform (str): Defaults to CUDA. Supports running on GPU for speed.
        first_residue (int | None): Defaults to None. If set, will restrict the 
            calculation to residues beginning with resid `first_residue`.
        last_residue (int | None): Defaults to None. If set, will restrict the 
            calculation to residues ending with resid `last_residue`.
    """
    def __init__(self, 
                 pdb: str, 
                 chain: str='A', 
                 platform: str='CUDA',
                 first_residue: Union[int, None]=None, 
                 last_residue: Union[int, None]=None):
        self.pdb = pdb
        self.chain = chain
        self.platform = Platform.getPlatformByName(platform)
        self.first = first_residue
        self.last = last_residue
        
    def get_system(self) -> System:
        """
        Builds implicit solvent OpenMM system.

        Returns:
            (System): OpenMM system object.
        """
        pdb = PDBFile(self.pdb)
        positions, topology = pdb.positions, pdb.topology
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        try:
            system = forcefield.createSystem(topology,
                                             soluteDielectric=1.,
                                             solventDielectric=80.)
        except ValueError:
            positions, topology = self.fix_pdb()
            system = forcefield.createSystem(topology,
                                             soluteDielectric=1.,
                                             solventDielectric=80.)

        self.positions = positions
        self.get_selection(topology)

        return system

    def compute(self, 
                positions: Union[np.ndarray, None]=None) -> None:
        """
        Compute interaction energy of system. Can optionally provide atomic
        positions such that this operation can be scaled onto a trajectory of
        frames rather than a static model.

        Arguments:
            positions (np.ndarray | None): Defaults to None. If provided, inject
                the positions into the OpenMM context.

        Returns:
            None
        """
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
    
    def get_selection(self, 
                      topology: Topology) -> None:
        """
        Using the poorly documented OpenMM selection language, get indices of
        atoms that we want to isolate for pairwise interaction energy calculation.

        Arguments:
            topology (Topology): OpenMM topology object.

        Returns:
            None
        """
        if self.first is None and self.last is None:
            selection = [a.index 
                        for a in topology.atoms() 
                        if a.residue.chain.id == self.chain]
        elif self.first is not None and self.last is None:
            selection = [a.index
                        for a in topology.atoms()
                        if a.residue.chain.id == self.chain 
                        and int(self.first) <= int(a.residue.id)]
        elif self.first is None:
            selection = [a.index
                        for a in topology.atoms()
                        if a.residue.chain.id == self.chain 
                        and int(self.last) >= int(a.residue.id)]
        else:
            selection = [a.index
                        for a in topology.atoms()
                        if a.residue.chain.id == self.chain 
                        and int(self.first) <= int(a.residue.id) <= int(self.last)]

        self.selection = selection

    def fix_pdb(self) -> None:
        """
        Using the OpenMM adjacent tool, PDBFixer, repair input PDB by adding
        hydrogens, and missing atoms such that we can actually construct an
        OpenMM system.

        Returns:
            None
        """
        fixer = PDBFixer(filename=self.pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        return fixer.positions, fixer.topology
    
    @property
    def interactions(self) -> np.ndarray:
        """
        Places the LJ and coulombic energies into an array of shape (1, 2).

        Returns:
            (np.ndarray): Energy array.
        """
        return np.vstack([self.lj, self.coulomb])

    @staticmethod
    def energy(context: Context, 
               solute_coulomb_scale: int=0, 
               solute_lj_scale: int=0, 
               solvent_coulomb_scale: int=0, 
               solvent_lj_scale: int=0) -> float:
        """
        Computes the potential energy for provided context object.

        Arguments:
            context (Context): OpenMM context object.
            solute_coulomb_scale (int): Defaults to 0. If 1 we will consider solute
                contributions to coulombic non-bonded energy.
            solute_lj_scale (int): Defaults to 0. If 1 we will consider solute
                contributions to LJ non-bonded energy.
            solvent_coulomb_scale (int): Defaults to 0. If 1 we will consider solvent
                contributions to coulombic non-bonded energy.
            solvent_lj_scale (int): Defaults to 0. If 1 we will consider solvent
                contributions to LJ non-bonded energy.

        Returns:
            (float): Computed energy term.
        """
        context.setParameter("solute_coulomb_scale", solute_coulomb_scale)
        context.setParameter("solute_lj_scale", solute_lj_scale)
        context.setParameter("solvent_coulomb_scale", solvent_coulomb_scale)
        context.setParameter("solvent_lj_scale", solvent_lj_scale)
        return context.getState(getEnergy=True, groups={0}).getPotentialEnergy()

class InteractionEnergyFrame(StaticInteractionEnergy):
    """
    Inherits from StaticInteractionEnergy and overloads `get_system` to allow for
    more easily running this analysis on a trajectory of frames. Requires the
    OpenMM system and topology to be built externally and passed in rather than
    beginning from a PDB file.

    Arguments:
        system (System): OpenMM system object.
        top (Topology): OpenMM topology object.
        chain (str): Defaults to A. The chain for which to compute the energy between.
            Computes energy between this chain and all other components in PDB file.
            Set to a whitespace if there are no chains in your PDB.
        platform (str): Defaults to CUDA. Supports running on GPU for speed.
        first_residue (int | None): Defaults to None. If set, will restrict the 
            calculation to residues beginning with resid `first_residue`.
        last_residue (int | None): Defaults to None. If set, will restrict the 
            calculation to residues ending with resid `last_residue`.
    """
    def __init__(self, 
                 system: System, 
                 top: Topology, 
                 chain: str='A', 
                 platform: str='CUDA',
                 first_residue: Union[int, None]=None, 
                 last_residue: Union[int, None]=None):
        super().__init__('', chain, platform, first_residue, last_residue)
        self.system = system
        self.top = top

    def get_system(self) -> System:
        """
        Sets self.selection via self.get_selection and returns existing OpenMM
        system object.

        Returns:
            (System): OpenMM system object.
        """
        self.get_selection(self.top)
        return self.system

class DynamicInteractionEnergy:
    """
    Class for obtaining interaction energies of a trajectory. Utilizes the 
    InteractionEnergyFrame child class to run per-frame energy calculations 
    and orchestrates the trajectory operations.

    Arguments:
        top (PathLike): Path to prmtop topology file.
        traj (PathLike): Path to DCD trajectory file.
        stride (int): Defaults to 1. The stride with which to move through the trajectory.
        chain (str): Defaults to A. The chain for which to compute the energy between.
            Computes energy between this chain and all other components in PDB file.
            Set to a whitespace if there are no chains in your PDB.
        platform (str): Defaults to CUDA. Supports running on GPU for speed.
        first_residue (int | None): Defaults to None. If set, will restrict the 
            calculation to residues beginning with resid `first_residue`.
        last_residue (int | None): Defaults to None. If set, will restrict the 
            calculation to residues ending with resid `last_residue`.
        progress_bar (bool): Defaults to False. If True a tqdm progress bar will
            display progress.
    """
    def __init__(self, 
                 top: PathLike, 
                 traj: PathLike, 
                 stride: int=1, 
                 chain: str='A', 
                 platform: str='CUDA',
                 first_residue: Union[int, None]=None,
                 last_residue: Union[int, None]=None,
                 progress_bar: bool=False):
        top = Path(top)
        traj = Path(traj)
        self.system = self.build_system(top)
        self.coordinates = self.load_traj(top, traj)
        self.stride = stride
        self.progress = progress_bar

        self.IE = InteractionEnergyFrame(self.system, self.top, chain, 
                                         platform, first_residue, last_residue)

    def compute_energies(self) -> None:
        """
        Computes the energy for each frame in trajectory, storing internally.

        Returns:
            None
        """
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
    
    def build_system(self, 
                     top: PathLike) -> System:
        """
        Builds OpenMM system for both pdb and prmtop topology files.

        Arguments:
            top (PathLike): Path to topology file.

        Returns:
            (System): OpenMM system object.
        """
        if top.suffix == '.pdb':
            top = PDBFile(str(top)).topology
            self.top = top
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            return forcefield.createSystem(top, 
                                           soluteDielectric=1., 
                                           solventDielectric=78.5)
        elif top.suffix == '.prmtop':
            top = AmberPrmtopFile(str(top))
            self.top = top
            return top.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                    nonbondedCutoff=2. * nanometers,
                                    constraints=HBonds)
        else:
            raise NotImplementedError(f'Error! Topology type {top} not implemented!')

    def load_traj(self, 
                  top: PathLike, 
                  traj: PathLike) -> np.ndarray:
        """
        Loads trajectory into mdtraj and extracts full coordinate array.

        Arguments:
            top (PathLike): Path to topology file.
            traj (PathLike): Path to trajectory file.

        Returns:
            (np.ndarray): Coordinate array of shape (n_frames, n_atoms, 3)
        """
        return md.load(str(traj), top=str(top)).xyz

    def setup_pbar(self) -> None:
        """
        Builds tqdm progress bar.

        Returns:
            None
        """
        self.pbar = tqdm(total=self.coordinates.shape[0], position=0, leave=False)
