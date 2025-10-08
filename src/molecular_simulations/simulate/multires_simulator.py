from ..build.build_calvados import CGBuilder
from ..build import ImplicitSolvent, ExplicitSolvent
from calvados import sim
from .omm_simulator import ImplicitSimulator, Simulator
from cg2all.script.convert_cg2all import main as convert
from openmm import *
from openmm.app import *
from openmm.unit import *
import pip._vendor.tomli as tomllib # for 3.10
from pathlib import Path
import os
from typing import Union, TypeVar

_T = TypeVar('_T')
OptPath = Union[Path, str, None]
PathLike = Union[Path, str]

@dataclass
class cg2all_defaults:
    standard_names: bool = True
    cg_model: str = 'ResidueBasedModel'
    device: str = 'cuda'
    n_proc: int = 1 # necessary for some reason

class MultiResolutionSimulator:
    """
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
                 cg2all_ckpt: OptPath = None):
        self.path = Path(path)
        self.input_pdb = input_pdb
        self.n_rounds = n_rounds
        self.cg_params = cg_params
        self.aa_params = aa_params
        self.cg2all_ckpt = cg2all_ckpt

    @classmethod
    def from_toml(cls: Type[_T], config: PathLike):
        """
        """
        with open(config, 'rb') as f:
            cfg = tomllib.load(f)
        settings = cfg['settings']
        cg_params = cfg['cg_params']
        aa_params = cfg['aa_params']
        path = settings['path']
        input_pdb = settings['input_pdb']
        n_rounds = settings['n_rounds']
        if 'cg2all_ckpt' in settings:
            cg2all_ckpt = settings['cg2all_ckpt']
        else:
            cg2all_ckpt = None
        
        return cls(path, 
                   input_pdb,
                   n_rounds, 
                   cg_params, 
                   aa_params, 
                   cg2all_ckpt = cg2all_ckpt)

    @staticmethod
    def cg_to_aa(input_pdb: PathLike, output_pdb: PathLike):
        convert_args = cg2all_defaults
        convert_args.in_pdb_fn = str(input_pdb)
        convert_args.outpdb_fn = str(output_pdb)
        convert(convert_args)

    def run_rounds(self):
        # restarts:
        #   need to check path for any half-finished runs
        #   e.g. if AA rounds 0,1,2 and CG rounds 0,1 are done, start with CG round 2

        for r in range(self.n_rounds):
            if r == 0:
                aa_path = self.path / f'aa_round{r}'

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
                    str(self.path / self.input_pdb),
                    protein = self.aa_params['protein'],
                    rna = self.aa_params['rna'],
                    dna = self.aa_params['dna'],
                    phos_protein = self.aa_params['phos_protein'],
                    use_amber = self.aa_params['use_amber'])
                
                aa_builder.build()

                aa_simulator = _aa_simulator(
                    aa_path,
                    equil_steps = self.aa_params['equilibration_steps'],
                    prod_steps = self.aa_params['production_steps'],
                    n_equil_cycles = 1)

                aa_simulator.run()

                # read openmm state file and output pdb
                aa_state = aa_simulator.simulation

                # convert AA to CG: calvados does this automatically

                # build CG
                cg_path = self.path / f'cg_round{r}'
                cg_params = self.cg_params
                cg_params['path'] = str(cg_path)
                cg_params['input_pdb'] = str(path/to/aa_sim) # TODO

                cg_builder = CGBuilder.from_dict(cg_params)
                cg_builder.build() # writes config and components yamls

                # run CG
                sim.run(path = str(cg_path), 
                        fconfig = str(cg_path / 'config.yaml'),
                        fcomponents = str(cg_path / 'components.yaml'))
            
                # convert CG to AA for next round


