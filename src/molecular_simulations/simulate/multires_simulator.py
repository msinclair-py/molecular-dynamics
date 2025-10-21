from ..build.build_calvados import CGBuilder
from ..build import ImplicitSolvent, ExplicitSolvent
from calvados import sim
from .omm_simulator import ImplicitSimulator, Simulator
from cg2all.script.convert_cg2all import main as convert
import openmm
from openmm.app import *
from openmm.unit import *
import subprocess
import parmed as pmd
import pip._vendor.tomli as tomllib # for 3.10
from pathlib import Path
import os
from typing import Union, Type, TypeVar

_T = TypeVar('_T')
OptPath = Union[Path, str, None]
PathLike = Union[Path, str]

class MultiResolutionSimulator:
    """
    Class for performing multi-resolution simulations with switching between CG and AA 
    representations. Utilizes CALVADOS for CG simulations and omm_simulator.py for AA
    simulations. 
    
    Arguments:
        path (PathLike): Path to simulation input files, also serves as output path.
        input_pdb (str): Input pdb for simulations, must exist in path.
        n_rounds (int): Number of rounds of CG/AA simulation to perform.
        cg_params (dict): Parameters for CG simulations. Initializes CGBuilder.
        aa_params (dict): Parameters for AA simulations. Initializes omm_simulator.
        cg2all_bin (str): Defaults to 'convert_cg2all'. Path to cg2all binary. Must
            be provided if cg2all is installed in a separate environment. 
        cg2all_ckpt (OptPath): Path to cg2all checkpoint file. 


    Usage:
        sim = MultiResolutionSimulator.from_toml('config.toml')
        sim.run()
    """
    def __init__(self, 
                 path: PathLike,
                 input_pdb: str,
                 n_rounds: int,
                 cg_params: dict, 
                 aa_params: dict,
                 cg2all_bin: str = 'convert_cg2all',
                 cg2all_ckpt: OptPath = None):
        self.path = Path(path)
        self.input_pdb = input_pdb
        self.n_rounds = n_rounds
        self.cg_params = cg_params
        self.aa_params = aa_params
        self.cg2all_bin = cg2all_bin
        self.cg2all_ckpt = cg2all_ckpt

    @classmethod
    def from_toml(cls: Type[_T], config: PathLike) -> _T:
        """
        Constructs MultiResolutionSimulator from .toml configuration file.
        Recommended method for instantiating MultiResolutionSimulator.
        """
        with open(config, 'rb') as f:
            cfg = tomllib.load(f)
        settings = cfg['settings']
        cg_params = cfg['cg_params'][0]
        aa_params = cfg['aa_params']
        path = settings['path']
        input_pdb = settings['input_pdb']
        n_rounds = settings['n_rounds']
        if 'cg2all_bin' in settings:
            cg2all_bin = settings['cg2all_bin']
        else:
            cg2all_bin = 'convert_cg2all'
        if 'cg2all_ckpt' in settings:
            cg2all_ckpt = settings['cg2all_ckpt']
        else:
            cg2all_ckpt = None
        
        return cls(path, 
                   input_pdb,
                   n_rounds, 
                   cg_params, 
                   aa_params, 
                   cg2all_bin = cg2all_bin,
                   cg2all_ckpt = cg2all_ckpt)

    @staticmethod
    def strip_solvent(simulation: Simulation,
                      output_pdb: PathLike = 'protein.pdb'):
        """
        Use parmed to strip solvent from an openmm simulation and write out pdb
        """
        struc = pmd.openmm.load_topology(
            simulation.topology,
            simulation.system,
            xyz = simulation.context.getState(getPositions=True).getPositions()
            )
        solvent_resnames = [
            'WAT', 'HOH', 'TIP3', 'TIP3P', 'SOL', 'OW', 'H2O',
            'NA', 'K', 'CL', 'MG', 'CA', 'ZN', 'MN', 'FE',
            'Na+', 'K+', 'Cl-', 'Mg2+', 'Ca2+', 'Zn2+', 'Mn2+', 'Fe2+', 'Fe3+',
            'SOD', 'POT', 'CLA'
            ]
        mask = ':' + ','.join(solvent_resnames)
        struc.strip(mask)
        struc.save(output_pdb)

    def run_rounds(self):
        """
        Main logic for running MultiResolutionSimulator
        """
        # restarts:
        #   need to check path for any half-finished runs
        #   e.g. if AA rounds 0,1,2 and CG rounds 0,1 are done, start with CG round 2

        for r in range(self.n_rounds):
            aa_path = self.path / f'aa_round{r}'
            aa_path.mkdir()

            if r == 0:
                input_pdb = str((self.path / self.input_pdb).resolve())
            else:
                input_pdb = str((self.path / f'cg_round{r-1}/last_frame.pdb').resolve())


            match self.aa_params['solvation_scheme']:
                case 'implicit':
                    _aa_builder = ImplicitSolvent
                    _aa_simulator = ImplicitSimulator
                case 'explicit':
                    _aa_builder = ExplicitSolvent
                    _aa_simulator = Simulator
                case _:
                    raise AttributeError("solvation_scheme must be 'implicit' or 'explicit'")

            aa_builder = _aa_builder(
                aa_path, 
                input_pdb,
                protein = self.aa_params['protein'],
                rna = self.aa_params['rna'],
                dna = self.aa_params['dna'],
                phos_protein = self.aa_params['phos_protein'],
                use_amber = self.aa_params['use_amber'],
                out = self.aa_params['out'])
            
            aa_builder.build()

            aa_simulator = _aa_simulator(
                aa_path,
                ff = 'amber',
                equil_steps = int(self.aa_params['equilibration_steps']),
                prod_steps = int(self.aa_params['production_steps']),
                n_equil_cycles = 1,
                device_ids = self.aa_params['device_ids'])

            aa_simulator.run()

            # strip solvent and output AA structure for next step (CG)
            self.strip_solvent(aa_simulator.simulation, 
                               str(aa_path / 'protein.pdb'))

            # build CG
            cg_path = self.path / f'cg_round{r}'
            cg_path.mkdir()
            cg_params = self.cg_params
            cg_params['config']['path'] = str(cg_path)
            cg_params['config']['input_pdb'] = str(aa_path / 'protein.pdb')

            cg_builder = CGBuilder.from_dict(cg_params)
            cg_builder.build() # writes config and components yamls

            # run CG
            sim.run(path = str(cg_path), 
                    fconfig = 'config.yaml',
                    fcomponents = 'components.yaml')
        
            # convert CG to AA for next round
            command = [self.cg2all_bin,
                       '-p', str(cg_path / 'top.pdb'),
                       '-d', str(cg_path / 'protein.dcd'),
                       '-o', str(cg_path / 'traj_aa.dcd'),
                       '-opdb', str(cg_path / 'last_frame.pdb'),
                       '--cg', 'ResidueBasedModel',
                       '--standard-name',
                       '--device', 'cuda',
                       '--proc', '1']
            if self.cg2all_ckpt is not None:
                command += ['--ckpt', self.cg2all_ckpt]

            result = subprocess.run(command, shell=False, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'cg2all error!\nstdout:{result.stdout}\nstderr:{result.stderr}')

