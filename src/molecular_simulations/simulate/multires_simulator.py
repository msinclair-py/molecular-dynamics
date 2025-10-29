from ..build.build_calvados import CGBuilder
from ..build import ImplicitSolvent, ExplicitSolvent
from calvados import sim
from .omm_simulator import ImplicitSimulator, Simulator
from cg2all.script.convert_cg2all import main as convert
import openmm
from openmm.app import *
from openmm.unit import *
import subprocess
import tempfile
import parmed as pmd
import pip._vendor.tomli as tomllib # for 3.10
from pathlib import Path
from dataclasses import dataclass
import os
from typing import Union, Type, TypeVar

_T = TypeVar('_T')
OptPath = Union[Path, str, None]
PathLike = Union[Path, str]

@dataclass
class sander_min_defaults:
    """
    Dataclass with default values for sander minimization.
    Creates the contents of a sander input file during init
    """
    imin=1       # Perform energy minimization
    maxcyc=5000  # Maximum number of minimization cycles
    ncyc=2500    # Switch from steepest descent to conjugate gradient after this many steps
    ntb=0        # Periodic boundary conditions (constant volume)
    ntr=0        # No restraints
    cut=10.0     # Nonbonded cutoff in Angstroms
    ntpr=10000   # Print energy every 10000 steps (don't print it)
    ntwr=5000    # Write restart file every 5000 steps (only once)
    ntxo=1       # Output restart file format (ASCII)

    def __init__(self):
        self.mdin_contents = f"""Minimization input
 &cntrl
  imin={self.imin},
  maxcyc={self.maxcyc},
  ncyc={self.ncyc},
  ntb={self.ntb},
  ntr={self.ntr},
  cut={self.cut:.1f},
  ntpr={self.ntpr},
  ntwr={self.ntwr},
  ntxo={self.ntxo} 
 /
 """

def sander_minimize(path: Path,
                    inpcrd_file: str,
                    prmtop_file: str,
                    sander_cmd: str) -> None:
    """
    Minimize MD system with sander and output new inpcrd file.
    
    Arguments:
        path (Path): Path to directory containing inpcrd and prmtop. New inpcrd will be
            written here as well.
        inpcrd_file (str): Name of inpcrd file in path
        prmtop_file (str): Name of prmtop file in path
        sander_cmd (str): Command for sander
    """
    defaults = sander_min_defaults()
    mdin = defaults.mdin_contents
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.in', dir=str(path)) as tmp_in:
        tmp_in.write(mdin)
        tmp_in.flush()
        outfile = Path(inpcrd_file).with_suffix('.min.inpcrd')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.out', dir=str(path)) as tmp_out:
            command = [sander_cmd, '-O', 
                       '-i', tmp_in.name, 
                       '-o', tmp_out.name,
                       '-p', str(path / prmtop_file), 
                       '-c', str(path / inpcrd_file),
                       '-r', str(path / outfile),
                       '-inf', str(path / 'min.mdinfo')] 
            result = subprocess.run(command, shell=False, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'sander error!\n{result.stderr}\n{result.stdout}')

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
        AMBERHOME (str | None): Defaults to None. Path to AMBERHOME (excluding bin). 
            Used for sander and pdb4amber. If None, assumes AmberTools binaries are 
            available in the current $PATH.

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
                 cg2all_ckpt: OptPath = None,
                 AMBERHOME: str | None = None):
        self.path = Path(path)
        self.input_pdb = input_pdb
        self.n_rounds = n_rounds
        self.cg_params = cg_params
        self.aa_params = aa_params
        self.cg2all_bin = cg2all_bin
        self.cg2all_ckpt = cg2all_ckpt
        self.AMBERHOME = Path(AMBERHOME) if AMBERHOME is not None else None

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

        if 'AMBERHOME' in settings:
            AMBERHOME = Path(settings['AMBERHOME'])
        else:
            AMBERHOME = None
        
        return cls(path, 
                   input_pdb,
                   n_rounds, 
                   cg_params, 
                   aa_params, 
                   cg2all_bin = cg2all_bin,
                   cg2all_ckpt = cg2all_ckpt,
                   AMBERHOME = AMBERHOME)

    @staticmethod
    def strip_solvent(simulation: Simulation,
                      output_pdb: PathLike = 'protein.pdb'
                      ) -> None:
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

    def run_rounds(self) -> None:
        """
        Main logic for running MultiResolutionSimulator.
        Does not currently handle restart runs (TODO).
        """

        for r in range(self.n_rounds):
            aa_path = self.path / f'aa_round{r}'
            aa_path.mkdir()

            if r == 0:
                input_pdb = str(self.path / self.input_pdb)
            else:
                input_pdb = str(self.path / f'cg_round{r-1}/last_frame.amber.pdb')


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
            
            # cg2all may create clashes which OpenMM minimization does not address.
            # Therefore, we want to minimize all cg2all-created structures with sander instead.
            if self.AMBERHOME is None:
                sander = 'sander'
            else:
                sander = str(self.AMBERHOME / 'bin/sander')
            sander_minimize(aa_path, 'system.inpcrd', 'system.prmtop', sander)

            aa_simulator = _aa_simulator(
                aa_path,
                coor_name = 'system.min.inpcrd',
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
        
            # convert CG to AA for next round using cg2all
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
                raise RuntimeError(f'cg2all error!\n{result.stderr}')

            # use pdb4amber to fix cg2all-generated pdb
            if self.AMBERHOME is None:
                command = ['pdb4amber'] 
            else:
                command = [str(self.AMBERHOME / 'bin/pdb4amber')]
            command += [str(cg_path / 'last_frame.pdb'), '-y']
            result = subprocess.run(command, shell=False, capture_output=True, text=True)
            if result.returncode == 0:
                with open(str(cg_path / 'last_frame.amber.pdb'), 'w') as f:
                    f.write(result.stdout)
            else:
                raise RuntimeError(f'pdb4amber error!\n{result.stderr}')
