import os
from calvados.cfg import Config, Job, Components
import numpy as np
import pip._vendor.tomli as tomllib # for 3.10
import yaml
from pathlib import Path
from typing import Any, Union, Type, TypeVar

_T = TypeVar('_T')
OptPath = Union[Path, str, None]
PathLike = Union[Path, str]

class CGBuilder:
    """
    Build CALVADOS system from pdb.
    Arguments:
        path (PathLike): Path to directory to contain run.
        input_pdb (PathLike): Path to input structure for simulation.
        residues_file (PathLike): Path to CALVADOS residues forcefield file.
        domains_file (PathLike): Path to file containing domain definitions.
        box_dim (list[float]): Dimensions (x,y,z) for orthonormal PBC.
        temp (float): Defaults to 310.0. Simulation temperature [K].
        ion_conc (float): Defaults to 0.15. Ion concentration [M].
        pH (float): Defaults to 7.4. pH.
        topol (str): Defaults to center. Varies initial placement of protein chains.
        dcd_freq (int): Defaults to 2000. Write frequency for output dcd.
        n_steps (int): Defaults to 10_000_000. Total number of steps to integrate
            during simulation (timestep of 10 fs).
        platform (str): Defaults to CUDA. OpenMM platform for running simulations.
        restart (str): Defaults to checkpoint. Style of OpenMM restart. 
        frestart (str): Defaults to restart.chk. Name of output restart files.
        verbose (bool): Defaults to True. Verbosity.
        molecule_type (str): Defaults to protein. Molecule type.
        nmol (int): Defaults to 1. Total number of molecules. 
        restraint (bool): Defaults to True. Whether to use secondary structure 
            restraints.
        charge_termini (str): Defaults to end-capped. Terminus patching. Allowed
            options are N, C, both, or the default.
        restraint_type (str): Defaults to harmonic. Type of restraint, either 
            harmonic or go.
        use_com (bool): Defaults to True. Where to apply restraintsi (com or CA). 
        colabfold (int): Defaults to 0. Predicted beta-column confidence style.
            Use 0 for AF and 1&2 for Colabfold.
        k_harmonic (float) Defaults to 700.0. Harmonic spring constant to use for
            restraints [kJ/mol].

    Usage:
        m = CGBuilder(*args)
        m.build()
    """
    def __init__(self,
                 path: PathLike,
                 input_pdb: PathLike,
                 residues_file: PathLike,
                 domains_file: PathLike,
                 box_dim: list[float],
                 temp: float = 310.,
                 ion_conc: float = 0.15,
                 pH: float = 7.4,
                 topol : str = 'center',
                 dcd_freq: int = 2000,
                 n_steps: int = 10_000_000,
                 platform: str = 'CUDA',
                 restart: str = 'checkpoint',
                 frestart: str = 'restart.chk',
                 verbose: bool = True,
                 molecule_type: str = 'protein',
                 nmol: int = 1,
                 restraint: bool = True, # secondary structure restraints
                 charge_termini: str = 'end-capped',
                 restraint_type: str = 'harmonic', # harmonic or go
                 use_com: bool = True, # apply to COMs or CAs
                 colabfold: int = 0, # (EBI AF=0, Colabfold=1&2)
                 k_harmonic: float = 700.):
        self.path = Path(path)
        self.input_pdb = Path(input_pdb)
        self.residues_file = Path(residues_file)
        self.domains_file = Path(domains_file)
        self.box_dim = box_dim
        self.temp = temp
        self.ion_conc = ion_conc
        self.pH = pH
        self.topol = topol
        self.dcd_freq = dcd_freq
        self.n_steps = n_steps
        self.platform = platform
        self.restart = restart
        self.frestart = frestart
        self.verbose = verbose
        self.molecule_type = molecule_type
        self.nmol = nmol
        self.restraint = restraint
        self.charge_termini = charge_termini
        self.restraint_type = restraint_type
        self.use_com = use_com
        self.colabfold = colabfold
        self.k_harmonic = k_harmonic

    @classmethod
    def from_dict(cls: Type[_T], cg_params: dict) -> _T:
        """
        Classmethod for creating CG simulation from toml config file. 
        Recommended method for instantiating CGBuilder.
        """
        conf_args = cg_params['config']
        comp_args = cg_params['components']
        
        path = Path(conf_args['path'])
        input_pdb = conf_args['input_pdb']
        residues_file = comp_args['residues_file']
        domains_file = comp_args['domains_file']
        box_dim = conf_args['box_dim']
        temp = conf_args['temp']
        ion_conc = conf_args['ion_conc']
        pH = conf_args['pH']
        topol = conf_args['topol']
        dcd_freq = conf_args['dcd_freq']
        n_steps = conf_args['n_steps']
        platform = conf_args['platform']
        restart = conf_args['restart']
        frestart = conf_args['frestart']
        verbose = conf_args['verbose']

        molecule_type = comp_args['molecule_type']
        nmol = comp_args['nmol']
        restraint = comp_args['restraint']
        charge_termini = comp_args['charge_termini']
        restraint_type = comp_args['restraint_type']
        use_com = comp_args['use_com']
        colabfold = comp_args['colabfold']
        k_harmonic = comp_args['k_harmonic']

        return cls(path,
                   input_pdb,
                   residues_file,
                   domains_file,
                   box_dim = box_dim,
                   temp = temp,
                   ion_conc = ion_conc,
                   pH = pH,
                   topol = topol,
                   dcd_freq = dcd_freq,
                   n_steps = n_steps,
                   platform = platform,
                   restart = restart,
                   frestart = frestart,
                   verbose = verbose,
                   molecule_type = molecule_type,
                   nmol = nmol,
                   restraint = restraint,
                   charge_termini = charge_termini,
                   restraint_type = restraint_type,
                   use_com = use_com,
                   colabfold = colabfold,
                   k_harmonic = k_harmonic)

    def build(self) -> None:
        """
        Prepares system for CALVADOS simulation. 
        Writes config and components yaml files.
        """
        self.write_config()
        self.write_components()
    
    def write_config(self) -> None:
        """
        Write CALVADOS config yaml.
        """
        config = Config(sysname = self.input_pdb.stem,
                        box = self.box_dim,
                        temp = self.temp,
                        ionic = self.ion_conc,
                        pH = self.pH,
                        topol = self.topol,
                        wfreq = self.dcd_freq,
                        steps = self.n_steps,
                        platform = self.platform,
                        restart = self.restart,
                        frestart = self.frestart,
                        verbose = self.verbose)

        with open(f'{self.path}/config.yaml','w') as f:
            yaml.dump(config.config, f)

    def write_components(self) -> None:
        """
        Write CALVADOS components yaml.
        """
        components = Components(molecule_type = self.molecule_type,
                                nmol = self.nmol,
                                restraint = self.restraint,
                                charge_termini = self.charge_termini,
                                fresidues = str(self.residues_file),
                                fdomains = str(self.domains_file),
                                pdb_folder = str(self.input_pdb.parent),
                                restraint_type = self.restraint_type,
                                use_com = self.use_com,
                                colabfold = self.colabfold,
                                k_harmonic = self.k_harmonic)
        components.add(name = self.input_pdb.stem)

        with open(f'{self.path}/components.yaml','w') as f:
            yaml.dump(components.components, f)


