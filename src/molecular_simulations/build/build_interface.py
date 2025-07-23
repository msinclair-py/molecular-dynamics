from .build_amber import ExplicitSolvent
import MDAnalysis as mda
import numpy as np
from pathlib import Path
from typing import Any, Union
import yaml

PathLike = Union[Path, str]
Config = dict[str, Any]

class InterfaceBuilder(ExplicitSolvent):
    def __init__(self, 
                 path: PathLike, 
                 pdb: str,
                 interfaces: Config,
                 target: PathLike,
                 binder: PathLike,
                 padding: float=10.,
                 protein: bool=True,
                 rna: bool=False,
                 dna: bool=False,
                 polarizable: bool=False):
        """
        For a given target/binder pair, build systems for driving binding to
        each of the supplied interfaces using DeepDriveMD. Includes writing
        out the required yaml files for running DeepDrive.
        """
        super().__init__(path, pdb, padding, protein, rna, dna, polarizable)
        self.interfaces = interfaces
        self.target = mda.Universe(target).select_atoms('all')
        self.binder = binder
        self.root = self.path / target.name[:-4] 
        self.com = self.target.center_of_mass()
    
    def build_all(self):
        """
        Iterates through each interface site for a given target and
        builds the corresponding system for the supplied miniprotein
        binder.
        """
        for site in self.interfaces.keys():
            # set pathing for this target/binder/site
            self.yaml_out = self.root / site / self.binder.name[:-4]
            self.out = self.yaml_out / 'ddmd' 
            self.out.mkdir(parents=True, exist_ok=True)
            self.out = self.out / 'system'
            self.build_dir = self.yaml_out / 'build'
            self.build_dir.mkdir(parents=True, exist_ok=True)
            self.pdb = self.build_dir / 'protein.pdb' # need full path to leverage parent methods

            cont_sel, dist_sel, vector, com, input_shape = self.parse_interface(site)

            binder = self.place_binder(np.array(vector, dtype=np.float32), 
                                       np.array(com, dtype=np.float32))
            self.merge_proteins(binder)

            self.path = self.build_dir
            self.build()
            
            self.write_ddmd_yaml(cont_sel, dist_sel)
            self.write_cvae_yaml(input_shape)

    def place_binder(self, 
                     vector: np.ndarray, 
                     com: np.ndarray) -> None:
        """
        Move binder nearby to the interface as defined by `vector`. Returns
        an MDAnalysis AtomGroup for the binder.
        """
        u = mda.Universe(self.binder)
        sel = u.select_atoms('all')
        binder_com = sel.center_of_mass()

        sel.positions -= com
        sel.positions += vector

        return sel

    def merge_proteins(self,
                       binder: mda.AtomGroup) -> None:
        """
        Merges the target and binder AtomGroups and writes out
        a unified PDB at `self.pdb` so as to leverage the existing
        pipeline for building explicit solvent systems.
        """
        merged_atoms = mda.Merge(self.target, binder)
        
        with mda.Writer(self.pdb) as W:
            W.write(merged_atoms)

    def parse_interface(self, 
                        site: str='site0') -> Config:
        """
        Returns the relevant data for the current interface site.
        """
        s = self.interfaces[site]
        N = len(s['contact_sel'][18:].split())
        inp_shape = (1, N, N)
        ret = [data for data in s.values()]
        ret.append(inp_shape)
        return ret # contact_sel, distance_sel, vector, com, inp_shape

    def write_ddmd_yaml(self,
                        contact_selection: str,
                        distance_selection: str) -> None:
        """
        Writes the simulation options yaml for a DeepDriveMD
        simulation.
        """
        yaml_settings = {
            'simulation_input_dir': 'ddmd',
            'num_workers': 4,
            'simulations_per_train': 6,
            'simulations_per_inference': 1,
            'num_total_simulations': 1000,
            'compute_settings': {
                'name': 'polaris',
                'num_nodes': 1,
                'worker_init': f'module use /soft/modulefiles; module load conda; \
                    conda activate deepdrive; cd {self.out}',
                'scheduler_options': '#PBS -l filesystems=home:eagle',
                'account': 'FoundEpidem',
                'queue': 'preemptable',
                'walltime': '72:00:00',
            },
            'simulation_settings': {
                'solvent_type': 'explicit',
                'dt_ps': 0.004,
                'mda_selection': contact_selection,
                'mda_selection_resid_list': None,
                'simulation_length_ns': 10,
                'report_interval_ps': 10,
                'temperature_kelvin': 300,
                'rmsd_reference_pdb': None,
                'distance_sels': distance_selection,
            },
            'train_settings': {
                'cvae_settings_yaml': 'cvae-prod-settings.yaml',
            },
        }

        with open(self.yaml_out / 'prod.yaml', 'w') as f:
            yaml.dump(yaml_settings, f)

    def write_cvae_yaml(self,
                        input_shape: list[int]) -> None:
        """
        Writes the CVAE options yaml for a DeepDriveMD
        simulation.
        """
        yaml_settings = {
            'input_shape': input_shape,
            'filters': [16, 16, 16, 16],
            'kernels': [3, 3, 3, 3],
            'strides': [1, 1, 1, 2],
            'affine_widths': [128],
            'affine_dropouts': [0.5],
            'latent_dim': 3,
            'lambda_rec': 1.0,
            'num_data_workers': 4,
            'prefetch_factor': 2,
            'batch_size': 64,
            'device': 'cuda',
            'optimizer_name': 'RMSprop',
            'optimizer_hparams': {
                'lr': 0.001,
                'weight_decay': 0.00001,
            },
            'epochs': 20,
            'checkpoint_log_every': 20,
            'plot_log_every': 20,
            'plot_n_samples': 5000,
            'plot_method': 'raw',
        }

        with open(self.yaml_out / 'cvae-prod-settings.yaml', 'w') as f:
            yaml.dump(yaml_settings, f)
