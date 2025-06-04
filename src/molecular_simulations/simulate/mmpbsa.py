from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
import polars as pl
import subprocess
from typing import Union

PathLike = Union[Path, str]

@dataclass
class MMPBSA_settings:
    top: PathLike
    dcd: PathLike
    selections: list[str]
    first_frame: int = 0
    last_frame: int = -1
    stride: int = 1
    out: str = 'mmpbsa'
    solvent_probe: float = 1.4
    offset: int = 0

class MMPBSA(MMPBSA_settings):
    """
    This is an experiment in patience. What follows is a reconstruction of the various
    pieces of code that run MM-P(G)BSA from AMBER but written in a more digestible manner
    with actual documentation. Herein we have un-CLI'd what should never have been a
    CLI and piped together the correct pieces of the ambertools ecosystem to perform
    MM-P(G)BSA and that alone. Your trajectory is required to be concatenated into a single
    continuous trajectory - or you can run this serially over each by instancing this class
    for each trajectory you have. In this way we have also disentangled the requirement to
    parallelize by use of MPI, allowing the user to choose their own parallelization/scaling
    scheme.

    Arguments:
        top (PathLike): Input topology for a solvated system. Should match the input trajectory.
        dcd (PathLike): Input trajectory. Can be DCD format or MDCRD already.
        fh (FileHandler): A helper class for performing file operations, including splitting
            out the various sub-topologies and sub-trajectories needed.
        selections (list[str]): A list of residue ID selections for the receptor and ligand
            in that order. Should be formatted for cpptraj (e.g. `:1-10`).
        first_frame (int): Defaults to 0. The first frame of the input trajectory to begin
            the calculations on.
        out (str): The prefix name for output files.
        solvent_probe (float): Defaults to 1.4Å. The probe radius to use for SA calculations.
        offset (int): Defaults to 0Å. I don't know what this does.
    """
    def __init__(self,
                 top: PathLike,
                 dcd: PathLike,
                 selections: list[str],
                 first_frame: int=0,
                 last_frame: int=-1,
                 stride: int=1,
                 out: str='mmpbsa',
                 solvent_probe: float=1.4,
                 offset: int=0,
                 **kwargs):
        super().__init__(top, dcd, selections, first_frame, last_frame, out, solvent_probe, offset)
        self.top = Path(self.top)
        self.traj = Path(self.dcd)

        self.fh = FileHandler(self.top, self.traj, self.selections, 
                              self.first_frame, self.last_frame, self.stride)
        self.analyzer = OutputAnalyzer(self.top.parent)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self) -> None:
        gb_mdin, pb_mdin = self.write_mdins()

        for (prefix, top, traj, pdb) in self.fh.files:
            self.calculate_sasa(prefix, top, traj)
            self.calculate_energy(prefix, top, traj, pdb, gb_mdin, 'gb')
            self.calculate_energy(prefix, top, traj, pdb, pb_mdin, 'pb')

        self.analyzer.parse_outputs()

    def calculate_sasa(self,
                       pre: str,
                       prm: PathLike,
                       trj: PathLike) -> None:
        """
        Runs the molsurf command in cpptraj to compute the SASA of a given system.
        """
        sasa = self.fh.path / 'sasa.in'
        sasa_in = [
            f'parm {prm}',
            f'trajin {trj}',
            f'molsurf :* out {pre}_surf.dat probe {self.solvent_probe} offset {self.offset}',
            'run',
            'quit'
        ]
        
        self.fh.write_file(sasa_in, sasa)
        subprocess.call(f'cpptraj -i {sasa}', shell=True)
        sasa.unlink()
    
    def calculate_energy(self,
                         pre: str,
                         prm: PathLike,
                         trj: PathLike,
                         pdb: PathLike, 
                         mdin: PathLike,
                         suf: str) -> None:
        """
        Runs mmpbsa_py_energy, an undocumented binary file which somehow computes the
        energy of a system.
        """
        subprocess.call(f'mmpbsa_py_energy -O -i {mdin} -p {prm} -c {pdb} -y {trj} -o {pre}_{suf}.mdout', shell=True)
    
    def write_mdins(self) -> tuple[Path, Path]:
        """
        Writes out the configuration files that are to be fed to mmpbsa_py_energy.
        These are also undocumented and I took the parameters from the location
        in which they are hardcoded in ambertools.
        """
        gb = self.fh.path / 'gb_mdin'
        gb_mdin = [
            'GB',
            'igb = 2',
            'extdiel = 78.3',
            'saltcon = 0.10',
            'surften = 0.0072',
            'rgbmax = 25.0'
        ]

        self.fh.write_file(gb_mdin, gb)

        pb = self.fh.path / 'pb_mdin'
        pb_mdin = [
            'PB',
            'inp = 2',
            'smoothopt = 1',
            'radiopt = 0',
            'npbopt = 0',
            'solvopt = 1',
            'maxitn = 1000',
            'nfocus = 2',
            'bcopt = 5',
            'eneopt = 2',
            'fscale = 8',
            'epsin = 1.0',
            'epsout = 80.0',
            'istrng = 0.10',
            'dprob = 1.4',
            'iprob = 2.0',
            'accept = 0.001',
            'fillratio = 4.0',
            'space = 0.5',
            'cutnb = 0',
            'sprob = 0.557',
            'cavity_surften = 0.0378',
            'cavity_offset = -0.5692'
        ]

        self.fh.write_file(pb_mdin, pb)

        return gb, pb


class OutputAnalyzer:
    def __init__(self, 
                 path: PathLike,
                 tolerance: float = 0.005):
        self.path = path
        self.tolerance = tolerance

        self.systems = ['receptor', 'ligand', 'complex']
        self.levels = ['gb', 'pb']

        self.solvent_contributions = ['EGB', 'ESURF', 'EPB', 'ECAVITY']

    def parse_outputs(self) -> None:
        """
        Parse all the output files.
        """        
        self.gb = pl.DataFrame()
        self.pb = pl.DataFrame()
        for system in self.systems:
            E_sasa = self.read_sasa(self.path / f'{system}_surf.dat')
            E_gb = self.read_GB(self.path / f'{system}_gb.mdout', system)
            E_pb = self.read_PB(self.path / f'{system}_pb.mdout', system)

            E_gb = E_gb.drop('ESURF').with_columns(E_sasa)

            self.gb = pl.concat([self.gb, E_gb], how='vertical')
            self.pb = pl.concat([self.pb, E_pb], how='vertical')
        
        all_cols = list(set(self.gb.columns + self.pb.columns))
        self.contributions = {
                'G gas': [col for col in all_cols
                          if col not in self.solvent_contributions], 
                'G solv': [col for col in all_cols
                          if col in self.solvent_contributions]
            }
        
        self.check_bonded_terms()
        self.generate_summary()
        self.compute_dG()

    def read_sasa(self,
                  _file: PathLike) -> np.ndarray:
        """
        Reads in the results of the cpptraj SASA calculation and returns the
        per-frame SASA scaled by a hardcoded value for surface tension that is
        not explained by MMPBSA
        """
        sasa = []
        for line in open(_file).readlines()[1:]:
            sasa.append(line.split()[-1].strip())

        return pl.Series('ESURF', np.array(sasa, dtype=float) * 0.0072)

    def read_GB(self,
                _file: PathLike,
                system: str) -> pl.DataFrame:
        """
        Read in the GB mdout files and returns a Polars dataframe of the values
        for each term for every frame. Also adds a `system` label to more easily
        compute summary statistics later.
        """
        gb_terms = ['BOND', 'ANGLE', 'DIHED', 'VDWAALS', 'EEL',
                    'EGB', '1-4 VDW', '1-4 EEL', 'RESTRAINT', 'ESURF']
        data = {gb_term: [] for gb_term in gb_terms}
        
        lines = open(_file, 'r').readlines()

        return self.parse_energy_file(lines, data, system)

    def read_PB(self,
                _file: PathLike,
                system: str) -> pl.DataFrame:
        """
        Read in the PB mdout files and returns a Polars dataframe of the values
        for each term for every frame. Also adds a `system` label to more easily
        compute summary statistics later.
        """
        pb_terms = ['BOND', 'ANGLE', 'DIHED', 'VDWAALS', 'EEL',
                    'EPB', '1-4 VDW', '1-4 EEL', 'RESTRAINT',
                    'ECAVITY', 'EDISPER']
        data = {pb_term: [] for pb_term in pb_terms}

        lines = open(_file, 'r').readlines()

        return self.parse_energy_file(lines, data, system)
    
    def parse_energy_file(self,
                          file_contents: list[str],
                          data: dict[str, list],
                          system: str) -> pl.DataFrame:
        """
        Parses 
        """
        idx = 0
        n_frames = 0
        while idx < len(file_contents):
            if file_contents[idx].startswith(' BOND'):
                for _ in range(4): # number of lines to read. DO NOT CHANGE!!!
                    line = file_contents[idx]
                    parsed = self.parse_line(line)
                    for key, val in parsed:
                        data[key].append(val)

                    idx += 1

            if 'Processing frame' in file_contents[idx]:
                n_frames = int(file_contents[idx].strip().split()[-1])

            idx +=1 

        data['system'] = [system] * n_frames
        
        return pl.DataFrame(
            {key: np.array(val) for key, val in data.items()}
        )

    def check_bonded_terms(self) -> None:
        bonded = ['BOND', 'ANGLE', 'DIHED', '1-4 VDW', '1-4 EEL']
        
        for theory_level in (self.gb, self.pb):
            a = theory_level.filter(pl.col('system') == 'receptor')
            b = theory_level.filter(pl.col('system') == 'ligand')
            c = theory_level.filter(pl.col('system') == 'complex')

            a = a.select(pl.col([col for col in a.columns if col in bonded])).to_numpy()
            b = b.select(pl.col([col for col in b.columns if col in bonded])).to_numpy()
            c = c.select(pl.col([col for col in c.columns if col in bonded])).to_numpy()

            diffs = np.array(c - b - a)
            if np.where(diffs >= self.tolerance)[0].size > 0:
                raise ValueError('Bonded terms for receptor + ligand != complex!')

        remove = ['RESTRAINT', 'EDISPER']
        self.gb = self.gb.select(
            pl.col([col for col in self.gb.columns if col not in remove])
        )
        self.pb = self.pb.select(
            pl.col([col for col in self.pb.columns if col not in remove])
        )

        self.n_frames = self.gb.height
        self.square_root_N = np.sqrt(self.n_frames)

    def generate_summary(self) -> None:
        full_statistics = {sys: {} for sys in self.systems}
        for theory, level in zip([self.gb, self.pb], self.levels):
            for system in self.systems:
                sys = theory.filter(pl.col('system') == system).drop('system')

                stats = {}
                for col in sys.columns:
                    mean = sys.select(pl.mean(col)).item()
                    stdev = sys.select(pl.std(col)).item()
                    
                    stats[col] = {'mean': mean, 
                                  'std': stdev, 
                                  'err': stdev / self.square_root_N}

                for energy, contributors in self.contributions.items():
                    pooled_data = sys.select(
                        pl.col([col for col in sys.columns if col in contributors])
                    ).to_numpy().flatten()

                    stats[energy] = {'mean': np.mean(pooled_data),
                                     'std': np.std(pooled_data),
                                     'err': np.std(pooled_data) / self.square_root_N}

                total_data = sys.to_numpy().flatten()
                stats['total'] = {'mean': np.mean(total_data),
                                  'std': np.std(total_data),
                                  'err': np.std(total_data) / self.square_root_N}

                full_statistics[system][level] = stats
        
        with open('statistics.json', 'w') as fout:
            json.dump(full_statistics, fout, indent=4)

    def compute_dG(self) -> None:
        differences = []
        for theory, level in zip([self.gb, self.pb], self.levels):
            diff_cols = [col for col in theory.columns if col != 'system']
            diff_arr = theory.filter(pl.col('system') == 'complex').drop('system').to_numpy()
            for system in self.systems[:2]:
                diff_arr -= theory.filter(pl.col('system') == system).drop('system').to_numpy()

            means = np.mean(diff_arr, axis=0)
            stds = np.std(diff_arr, axis=0)
            errs = stds / self.square_root_N

            gas_solv_phase = []
            for energy, contributors in self.contributions.items():
                indices = [i for i, diff_col in enumerate(diff_cols) 
                           if diff_col in contributors]
                contribution = np.sum(diff_arr[:, indices], axis=1)
                gas_solv_phase.append(contribution)

                diff_cols.append(energy)
                means = np.concatenate((means, [np.mean(contribution)]))
                stds = np.concatenate((stds, [np.std(contribution)]))
                errs = np.concatenate((errs, [np.std(contribution) / self.square_root_N]))
            
            diff_cols.append('Total')
            total = np.sum(np.vstack(gas_solv_phase), axis=0)
            
            means = np.concatenate((means, [np.mean(total)]))
            stds = np.concatenate((stds, [np.std(total)]))
            errs = np.concatenate((errs, [np.std(total) / self.square_root_N]))

            data = np.vstack((means, stds, errs))
            
            differences.append(pl.DataFrame(
                {diff_cols[i]: data[:,i] for i in range(len(diff_cols))}
            ))

        self.pretty_print(differences)

    def pretty_print(self,
                     dfs: list[pl.DataFrame]) -> None:
        print_statement = []
        for df, level in zip(dfs, ['Generalized Born ', 'Poisson Boltzmann']):
            print_statement += [
                f'{" ":<20}=========================',
                f'{" ":<20}=== {level} ===',
                f'{" ":<20}=========================',
                'Energy Component    Average         Std. Dev.       Std. Err. of Mean',
                '---------------------------------------------------------------------'
            ]
            for col in df.columns:
                mean, std, err = [x.item() for x in df.select(pl.col(col)).to_numpy()]
                if abs(mean) <= self.tolerance:
                    continue

                print_statement.append(f'{col:<20}{mean:<16.3f}{std:<16.3f}{err:<16.3f}')

            print_statement += ['']
        
        print_statement = '\n'.join(print_statement)
        with open('deltaG.txt', 'w') as fout:
            fout.write(print_statement)
        
        print(print_statement)

    @staticmethod
    def parse_line(line) -> tuple[list[str], list[float]]:
        eq_split = line.split('=')
        
        if len(eq_split) == 2:
            splits = [eq_spl.strip() for eq_spl in eq_split]
        else:
            splits = [eq_split[0].strip()]

            for i in range(1, len(eq_split) - 1):
                splits += [spl.strip() for spl in eq_split[i].strip().split('  ')]

            splits += [eq_split[-1].strip()]
        
        keys = splits[::2]
        vals = np.array(splits[1::2], dtype=float)
        
        return zip(keys, vals)


class FileHandler:
    def __init__(self,
                 top: Path,
                 traj: Path,
                 sels: list[str],
                 first: int,
                 last: int,
                 stride: int):
        self.top = top
        self.traj = traj
        self.selections = sels
        self.ff = first
        self.lf = last
        self.stride = stride
        
        self.path = self.top.parent / 'mmpbsa'
        self.path.mkdir(exist_ok=True)

        self.prepare_topologies()
        self.prepare_trajectories()

    def prepare_topologies(self) -> None:
        """
        Slices out each sub-topology for the desolvated complex, receptor and
        ligand using cpptraj due to the difficulty of working with AMBER FF
        files otherwise (including PARMED).
        """
        self.topologies = [
            self.path / 'complex.prmtop',
            self.path / 'receptor.prmtop',
            self.path / 'ligand.prmtop'
        ]
        
        cpptraj_in = [
            f'parm {self.top}',
            'parmstrip :Na+,Cl-,WAT',
            'parmbox nobox',
            f'parmwrite out {self.topologies[0]}',
            'run',
            'clear all',
            f'parm {self.topologies[0]}',
            f'parmstrip {self.selections[0]}',
            f'parmwrite out {self.topologies[1]}',
            'run',
            'clear all',
            f'parm {self.topologies[0]}',
            f'parmstrip {self.selections[1]}',
            f'parmwrite out {self.topologies[2]}',
            'run',
            'quit'
        ]
        
        script = self.path  / 'cpptraj.in'
        self.write_file('\n'.join(cpptraj_in), script)
        subprocess.call(f'cpptraj -i {script}', shell=True)
        script.unlink()
        
    def prepare_trajectories(self) -> None:
        """
        Converts DCD trajectory to AMBER CRD format which is explicitly
        required by MM-G(P)BSA.
        """
        self.trajectories = [path.with_suffix('.crd') for path in self.topologies]
        self.pdbs = [path.with_suffix('.pdb') for path in self.topologies]

        frame_control = f'start {self.ff} stop {self.lf} offset {self.stride}'
        
        if self.traj.with_suffix('.crd').exists():
            cpptraj_in = []
        else:
            cpptraj_in = [
                f'parm {self.top}', 
                f'trajin {self.traj}',
                f'trajout {self.traj.with_suffix(".crd")} crd {frame_control}',
                'run',
                'clear all',
            ]

        self.traj = self.traj.with_suffix('.crd')

        cpptraj_in += [
            f'parm {self.top}', 
            f'trajin {self.traj}',
            'strip :WAT,Na+,Cl*',
            'autoimage',
            f'rmsd !(:WAT,Cl*,CIO,Cs+,IB,K*,Li+,MG*,Na+,Rb+,CS,RB,NA,F,CL) mass first',
            f'trajout {self.trajectories[0]} crd nobox',
            f'trajout {self.pdbs[0]} pdb onlyframes 1',
            'run',
            'clear all',
            f'parm {self.topologies[0]}', 
            f'trajin {self.trajectories[0]}',
            f'strip {self.selections[0]}',
            f'trajout {self.trajectories[1]} crd',
            f'trajout {self.pdbs[1]} pdb onlyframes 1',
            'run',
            'clear all',
            f'parm {self.topologies[0]}', 
            f'trajin {self.trajectories[0]}',
            f'strip {self.selections[1]}',
            f'trajout {self.trajectories[2]} crd',
            f'trajout {self.pdbs[2]} pdb onlyframes 1',
            'run',
            'quit'
        ]

        name = self.path / 'mdcrd.in'
        self.write_file('\n'.join(cpptraj_in), name)
        subprocess.call(f'cpptraj -i {name}', shell=True)

        name.unlink()

    @property
    def files(self) -> tuple[list[str]]:
        _order = [self.path / prefix for prefix in ['complex', 'receptor', 'ligand']]
        return zip(_order, self.topologies, self.trajectories, self.pdbs)

    @staticmethod
    def write_file(lines: list[str],
                   filepath: PathLike) -> None:
        if isinstance(lines, list):
            lines = '\n'.join(lines)
        with open(str(filepath), 'w') as f:
            f.write(lines)
