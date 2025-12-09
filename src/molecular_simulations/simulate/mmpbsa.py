from concurrent.futures import as_completed, ThreadPoolExecutor, wait, ALL_COMPLETED
from dataclasses import dataclass
import json
import logging
import os
import pandas as pd
from pathlib import Path
import polars as pl
import re
import subprocess
import time
from typing import Literal, Optional, Union
        
# This is simply to enable higher level parallelism by parsl/academy
# Numpy by default allows all threads to be used and in agentic settings
# we have seen oversubscription of threads and some calculations fail to
# to write out. These settings must be set BEFORE importing numpy
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

PathLike = Union[Path, str]

logger = logging.getLogger(__name__)

def _run_energy_calculation(args: tuple[str],
                            max_retries: int=3) -> tuple[Path, bool, str]:
    """Worker function for parallel energy calculations.
    Must be module level for ThreadPoolExecutor pickling.

    Args:
        args (tuple): Tuple of (mmpbsa_binary, mdin_path, prmtop, pdb, traj_chunk, output_path, cwd)
        max_retries (int): Number of retry attempts for failed calculations

    Returns:
        (Path): Path to the output file.
    """
    mmpbsa_binary, mdin, prm, pdb, trj, out, cwd = args
    cmd = f'{mmpbsa_binary} -O -i {mdin} -p {prm} -c {pdb} -y {trj} -o {out}'

    expected_output = Path(cwd) / out

    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, shell=True, cwd=str(cwd), 
                                    capture_output=True, text=True)

            if expected_output.exists() and expected_output.stat().st_size > 0:
                with open(expected_output, 'r') as f:
                    content = f.read()
                    if ' BOND' in content:
                        return (out, True, '')
                    else:
                        error = 'Output file exists but contains no energy data'
            else:
                error = 'Output file missing or empty after subprocess complete'

            if result.returncode != 0:
                error = f'Return code: {result.returncode}: {result.stderr or result.stdout}'

        except subprocess.TimeoutExpired:
            error = 'Calculation timed out'
        except Exception as e:
            error = f'Exception: {e}'

        if attempt < max_retries - 1:
            logger.warning(f'Energy calculation {out} failed (attempt {attempt + 1}/{max_retries})')
            logger.warning(f'Error: {error}')
            time.sleep(2 ** attempt)

    return (out, False, error)

def _run_sasa_calculation(args: tuple[str],
                          max_retries: int=3) -> tuple[Path, bool, str]:
    """Worker function for parallel SASA calculations.
    Must be module level for ThreadPoolExecutor pickling."""
    cpptraj_binary, sasa_script, cwd = args

    # Parse expected output from script
    script_path = Path(sasa_script)
    with open(script_path, 'r') as f:
        script_content = f.read()

    match = re.search(r'molsurf\s+.*?\s+out\s+(\S+)', script_content)
    if match:
        expected_output = Path(cwd) / match.group(1)
    else:
        expected_output = None
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(f'{cpptraj_binary} -i {sasa_script}', shell=True, cwd=str(cwd),
                                    capture_output=True, text=True)
            if expected_output and expected_output.exists():
                if expected_output.stat().st_size > 0:
                    with open(expected_output, 'r') as f:
                        lines = f.readlines()
                        data_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                        if len(data_lines) > 0:
                            return (sasa_script, True, '')
                        else:
                            error = 'Output file has no data lines'
                else: 
                    error = 'Expected output file is empty'
            else:
                error = f'Expected output file not found: {expected_output}'

            if result.returncode != 0:
                error = f'Return code {result.returncode}: {result.stderr or result.stdout}'

        except subprocess.TimeoutExpired:
            error = 'Calculation timed out'
        except Exception as e:
            error = f'Exception: {e}'

        if attempt < max_retries - 1:
            logger.warning(f'SASA calculation {sasa_script} failed (attempt {attempt+1}/{max_retries})')
            time.sleep(2 ** attempt)

    return (script_path, False, error)

@dataclass
class MMPBSA_settings:
    top: PathLike
    dcd: PathLike
    selections: list[str]
    first_frame: int = 0
    last_frame: int = -1
    stride: int = 1
    n_cpus: int = 1
    out: str = 'mmpbsa'
    solvent_probe: float = 1.4
    offset: int = 0
    gb_surften: float=0.0072
    gb_surfoff: float=0.

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
        selections (list[str]): A list of residue ID selections for the receptor and ligand
            in that order. Should be formatted for cpptraj (e.g. `:1-10`).
        first_frame (int): Defaults to 0. The first frame of the input trajectory to begin
            the calculations on.
        last_frame (int): Defaults to -1. Optional final frame to cut trajectory at. If -1,
            acts as a flag to run the whole trajectory.
        stride (int): Defaults to 1. The number of frames to stride the trajectory by.
        n_cpus (int): Number of parallel processes
        out (str): The prefix name or path for output files.
        solvent_probe (float): Defaults to 1.4Å. The probe radius to use for SA calculations.
        offset (int): Defaults to 0Å. I don't know what this does.
        gb_surften (float): Defaults to 0.0072.
        gb_surfoff (float): Defaults to 0.0.
        parallel_mode (str): 'frame' for frame-level parallelization (recommended),
                             'system' for system-level parallelization,
                             'hybrid' for both (most aggressive)
    """
    def __init__(self,
                 top: PathLike,
                 dcd: PathLike,
                 selections: list[str],
                 first_frame: int=0,
                 last_frame: int=-1,
                 stride: int=1,
                 n_cpus: int=1,
                 out: str='mmpbsa',
                 solvent_probe: float=1.4,
                 offset: int=0,
                 gb_surften: float=0.0072,
                 gb_surfoff: float=0.,
                 amberhome: Optional[str]=None,
                 parallel_mode: Literal['frame', 'serial'] = 'frame',
                 **kwargs):
        super().__init__(top=top, 
                         dcd=dcd, 
                         selections=selections, 
                         first_frame=first_frame, 
                         last_frame=last_frame, 
                         stride=stride, 
                         n_cpus=n_cpus,
                         out=out, 
                         solvent_probe=solvent_probe, 
                         offset=offset, 
                         gb_surften=gb_surften, 
                         gb_surfoff=gb_surfoff)
        self.parallel_mode = parallel_mode
        self.top = Path(self.top).resolve()
        self.traj = Path(self.dcd).resolve()
        self.path = self.top.parent
        if out == 'mmpbsa':
            self.path = self.path / 'mmpbsa'
        else:
            self.path = Path(out).resolve()

        self.path.mkdir(exist_ok=True, parents=True)

        self.cpptraj = 'cpptraj'
        self.mmpbsa_py_energy = 'mmpbsa_py_energy'
        if amberhome is None: # we are overriding AMBERHOME or using another env's install
            if 'AMBERHOME' in os.environ:
                amberhome = os.environ['AMBERHOME']
            else:
                raise ValueError('AMBERHOME not set in env vars!')
        
        self.cpptraj = Path(amberhome) / 'bin' / self.cpptraj
        self.mmpbsa_py_energy = Path(amberhome) / 'bin' / self.mmpbsa_py_energy

        self.fh = FileHandler(
            top=self.top, 
            traj=self.traj, 
            path=self.path, 
            sels=self.selections, 
            first=self.first_frame, 
            last=self.last_frame, 
            stride=self.stride,
            cpptraj_binary=self.cpptraj,
            n_chunks=self.n_cpus
        )

        self.analyzer = OutputAnalyzer(
            path=self.path, 
            surface_tension=self.gb_surften, 
            sasa_offset=self.gb_surfoff
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
            

    def run(self) -> None:
        """
        Main logic of MM-PBSA with parallelization.
        
        Depending on parallel_mode:
        - 'frame': Splits trajectory into chunks, processes in parallel
        """
        logger.debug(f'Preparing MM-PBSA calculation with {self.n_cpus} CPUs (mode: {self.parallel_mode})')
        gb_mdin, pb_mdin = self.write_mdins()

        if self.parallel_mode == 'frame':
            self._run_frame_parallel(gb_mdin, pb_mdin)
        else:
            # Fallback to serial
            self._run_serial(gb_mdin, pb_mdin)

        logger.debug('Collating results.')
        self.analyzer.parse_outputs()

        self.free_energy = self.analyzer.free_energy

    def _run_serial(self, gb_mdin: Path, pb_mdin: Path) -> None:
        """Original serial implementation."""
        for (prefix, top, traj, pdb) in self.fh.files:
            logger.debug(f'Computing energy terms for {prefix.name}.')
            self.calculate_sasa(prefix, top, traj)
            self.calculate_energy(prefix, top, traj, pdb, gb_mdin, 'gb')
            self.calculate_energy(prefix, top, traj, pdb, pb_mdin, 'pb')

    def _run_frame_parallel(self, gb_mdin: Path, pb_mdin: Path) -> None:
        """
        Frame-level parallelization: split trajectory into chunks, process in parallel.
        This provides the best speedup for long trajectories.
        """
        # Collect all calculation tasks
        energy_tasks = []
        sasa_tasks = []
        
        for (prefix, top, traj_chunks, pdb) in self.fh.files_chunked:
            system_name = prefix.name
            logger.debug(f'Preparing parallel energy calculations for {system_name}.')
            
            # SASA calculations for each chunk
            for i, traj_chunk in enumerate(traj_chunks):
                sasa_script = self._write_sasa_script(prefix, top, traj_chunk, chunk_idx=i)
                sasa_tasks.append((str(self.cpptraj), str(sasa_script), str(self.path)))
            
            # Energy calculations for each chunk
            for i, traj_chunk in enumerate(traj_chunks):
                # GB calculation
                out_gb = f'{system_name}_chunk{i}_gb.mdout'
                energy_tasks.append((
                    str(self.mmpbsa_py_energy), str(gb_mdin), str(top), 
                    str(pdb), str(traj_chunk), out_gb, str(self.path)
                ))
                # PB calculation
                out_pb = f'{system_name}_chunk{i}_pb.mdout'
                energy_tasks.append((
                    str(self.mmpbsa_py_energy), str(pb_mdin), str(top),
                    str(pdb), str(traj_chunk), out_pb, str(self.path)
                ))
        
        # Run SASA calculations in parallel
        logger.debug(f'Running {len(sasa_tasks)} SASA calculations in parallel.')
        sasa_failures = []
        with ThreadPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = []
            for task in sasa_tasks:
                futures.append(executor.submit(_run_sasa_calculation, task))

            logger.debug(f'Submitted {len(futures)} SASA futures, waiting for completion...')

            # Wait for ALL to complete before proceeding
            done, _ = wait(futures, return_when=ALL_COMPLETED)

            for future in done:
                script, success, error = future.result()
                if not success:
                    sasa_failures.append((script, error))
                    logger.error(f'SASA calculation failed: {script}: {error[:300]}')

        if sasa_failures:
            failed_scripts = [f[0] for f in sasa_failures]
            raise RuntimeError(f'{len(sasa_failures)} SASA calculations failed: {failed_scripts}')  
        logger.debug('All SASA calculations completed successfully')

        # Combine SASA results
        self._combine_sasa_chunks()
        
        # Run Energy calculations in parallel
        logger.debug(f'Running {len(energy_tasks)} energy calculations in parallel.')
        energy_failures = []
        with ThreadPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = []
            for task in energy_tasks:
                futures.append(executor.submit(_run_energy_calculation, task))

            logger.debug(f'Submitted {len(futures)} Energy futures, waiting for completion...')

            # Wait for ALL to complete before proceeding
            done, _ = wait(futures, return_when=ALL_COMPLETED)

            for future in done:
                script, success, error = future.result()
                if not success:
                    energy_failures.append((script, error))
                    logger.error(f'Energy calculation failed: {script}: {error[:300]}')

        if energy_failures:
            failed_scripts = [f[0] for f in energy_failures]
            raise RuntimeError(f'{len(energy_failures)} Energy calculations failed: {failed_scripts}')  
        logger.debug('All Energy calculations completed successfully')

        # Combine Energy results
        self._combine_energy_chunks()

        self._verify_combined_outputs()

    def _write_sasa_script(self, prefix: Path, prm: Path, trj: Path, 
                          chunk_idx: int = 0) -> Path:
        """Write a SASA calculation script for a trajectory chunk."""
        sasa = self.path / f'sasa_{prefix.name}_chunk{chunk_idx}.in'
        out_file = f'{prefix.name}_chunk{chunk_idx}_surf.dat'
        sasa_in = [
            f'parm {prm}',
            f'trajin {trj}',
            f'molsurf :* out {out_file} probe {self.solvent_probe} offset {self.offset}',
            'run',
            'quit'
        ]
        self.fh.write_file(sasa_in, sasa)
        return sasa

    def _combine_sasa_chunks(self) -> None:
        """Combine SASA results from all chunks into single files (in correct frame order)."""
        def extract_chunk_idx(filepath: Path) -> int:
            """Extract chunk index from filename for proper numerical sorting."""
            match = re.search(r'_chunk(\d+)_', filepath.name)
            return int(match.group(1)) if match else 0
        
        for system in ['complex', 'receptor', 'ligand']:
            combined_data = []
            chunk_files = list(self.path.glob(f'{system}_chunk*_surf.dat'))
            # Sort numerically by chunk index, not lexicographically
            chunk_files.sort(key=extract_chunk_idx)
            
            for chunk_file in chunk_files:
                with open(chunk_file) as f:
                    lines = f.readlines()
                    if combined_data:
                        # Skip header for subsequent chunks
                        combined_data.extend(lines[1:])
                    else:
                        combined_data.extend(lines)
            
            # Write combined file
            output = self.path / f'{system}_surf.dat'
            with open(output, 'w') as f:
                f.writelines(combined_data)
            
            # Clean up chunk files
            for chunk_file in chunk_files:
                chunk_file.unlink()

    def _combine_energy_chunks(self) -> None:
        """Combine energy results from all chunks into single files (in correct frame order)."""
        def extract_chunk_idx(filepath: Path) -> int:
            """Extract chunk index from filename for proper numerical sorting."""
            match = re.search(r'_chunk(\d+)_', filepath.name)
            return int(match.group(1)) if match else 0
        
        for system in ['complex', 'receptor', 'ligand']:
            for level in ['gb', 'pb']:
                combined_data = []
                chunk_files = list(self.path.glob(f'{system}_chunk*_{level}.mdout'))
                # Sort numerically by chunk index, not lexicographically
                chunk_files.sort(key=extract_chunk_idx)
                
                for chunk_file in chunk_files:
                    with open(chunk_file) as f:
                        content = f.read()
                        combined_data.append(content)
                
                # Write combined file (mdout format allows concatenation of frame data)
                output = self.path / f'{system}_{level}.mdout'
                with open(output, 'w') as f:
                    f.write('\n'.join(combined_data))
                
                # Clean up chunk files
                for chunk_file in chunk_files:
                    chunk_file.unlink()

    def calculate_sasa(self,
                       pre: str,
                       prm: PathLike,
                       trj: PathLike) -> None:
        """
        Runs the molsurf command in cpptraj to compute the SASA of a given system.

        Arguments:
            pre (str): Prefix for output SASA file.
            prm (PathLike): Path to prmtop file.
            trj (PathLike): Path to CRD trajectory file.

        Returns:
            None
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

        subprocess.run(f'{self.cpptraj} -i {sasa}', shell=True, cwd=str(self.path),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sasa.unlink()
    
    def calculate_energy(self,
                         pre: str,
                         prm: PathLike,
                         trj: PathLike,
                         pdb: PathLike, 
                         mdin: PathLike,
                         suf: str) -> None:
        """
        Runs mmpbsa_py_energy, an undocumented binary file which somehow mysteriously 
        computes the energy of a system. This software is not only undocumented but is
        a binary which we cannot inspect ourselves.
        
        Arguments:
            pre (str): Prefix for output file.
            prm (PathLike): Path to prmtop file.
            trj (PathLike): Path to CRD trajectory file.
            pdb (PathLike): Path to PDB file.
            mdin (PathLike): Configuration file for the program.
            suf (str): Suffix for output file.

        Returns:
            None
        """
        cmd = f'{self.mmpbsa_py_energy} -O -i {mdin} -p {prm} -c {pdb} -y {trj} -o {pre}_{suf}.mdout'
        subprocess.run(cmd, shell=True, cwd=str(self.path), 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def write_mdins(self) -> tuple[Path, Path]:
        """
        Writes out the configuration files that are to be fed to mmpbsa_py_energy.
        These are also undocumented and I took the parameters from the location
        in which they are hardcoded in ambertools.

        Returns:
            (tuple[Path, Path]): Tuple of paths to the GB mdin and the PB mdin.
        """
        gb = self.fh.path / 'gb_mdin'
        gb_mdin = [
            'GB',
            'igb = 2',
            'extdiel = 78.3',
            'saltcon = 0.10',
            f'surften = {self.gb_surften}',
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

    def _verify_combined_outputs(self) -> None:
        """Verify all required output files exist and have valid content.
        Raises RuntimeError if any files are missing or invalid"""
        missing_files = []
        empty_files = []
        invalid_files = []
        
        for system in ['complex', 'receptor', 'ligand']:
            # Check SASA file
            sasa_file = self.path / f'{system}_surf.dat'
            if not sasa_file.exists():
                missing_files.append(str(sasa_file))
            elif sasa_file.stat().st_size == 0:
                empty_files.append(str(sasa_file))
            else:
                # Verify it has data lines
                with open(sasa_file) as f:
                    data_lines = [l for l in f if l.strip() and not l.strip().startswith('#')]
                    if len(data_lines) == 0:
                        invalid_files.append(f'{sasa_file} (no data lines)')
            
            # Check energy files
            for level in ['gb', 'pb']:
                energy_file = self.path / f'{system}_{level}.mdout'
                if not energy_file.exists():
                    missing_files.append(str(energy_file))
                elif energy_file.stat().st_size == 0:
                    empty_files.append(str(energy_file))
                else:
                    # Verify it has BOND lines (energy data)
                    with open(energy_file) as f:
                        content = f.read()
                        if ' BOND' not in content:
                            invalid_files.append(f'{energy_file} (no energy data)')
        
        errors = []
        if missing_files:
            errors.append(f"Missing files: {missing_files}")
        if empty_files:
            errors.append(f"Empty files: {empty_files}")
        if invalid_files:
            errors.append(f"Invalid files: {invalid_files}")
        
        if errors:
            raise RuntimeError(f"Output verification failed: {'; '.join(errors)}")


class OutputAnalyzer:
    """
    Analyzes the outputs from an MM-PBSA run. Stores data in a Polars dataframe
    internally, and writes out data in the form of json/plain text.
    """
    def __init__(self, 
                 path: PathLike,
                 surface_tension: float=0.0072,
                 sasa_offset: float=0.,
                 _tolerance: float = 0.005,
                 log: bool=True):
        self.path = Path(path)
        self.surften = surface_tension
        self.offset = sasa_offset
        self.tolerance = _tolerance
        self.log = log

        self.free_energy = None

        self.systems = ['receptor', 'ligand', 'complex']
        self.levels = ['gb', 'pb']

        self.solvent_contributions = ['EGB', 'ESURF', 'EPB', 'ECAVITY']

    def parse_outputs(self) -> None:
        """
        Parse all the output files.

        Returns:
            None
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
        a mostly undocumented heuristic.

        Arguments:
            _file (PathLike): Path to a file containing the SASA data.

        Returns:
            (np.ndarray): A numpy array of the per-frame rescaled SASA energies.
        """
        df = pd.read_csv(_file, sep='\s+') # read in dataframe
        sasa = df.iloc[:, -1].to_numpy(dtype=float) * self.surften + self.offset

        return pl.Series('ESURF', sasa)

    def read_GB(self,
                _file: PathLike,
                system: str) -> pl.DataFrame:
        """
        Read in the GB mdout files and returns a Polars dataframe of the values
        for each term for every frame. Also adds a `system` label to more easily
        compute summary statistics later.

        Arguments:
            _file (PathLike): Energy data file path.
            system (str): String label for which system we are processing (e.g. complex).

        Returns:
            (pl.DataFrame): Polars dataframe containing the parsed energy data.
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

        Arguments:
            _file (PathLike): Energy data file path.
            system (str): String label for which system we are processing (e.g. complex).

        Returns:
            (pl.DataFrame): Polars dataframe containing the parsed energy data.
        """
        pb_terms = ['BOND', 'ANGLE', 'DIHED', 'VDWAALS', 'EEL',
                    'EPB', '1-4 VDW', '1-4 EEL', 'RESTRAINT',
                    'ECAVITY', 'EDISPER']
        data = {pb_term: [] for pb_term in pb_terms}

        lines = open(_file, 'r').readlines()

        return self.parse_energy_file(lines, data, system)
    
    def parse_energy_file(self, file_contents: list[str],
                         data: dict[str, list], system: str) -> pl.DataFrame:
        """Parse energy file contents."""
        for line in file_contents:
            if '=' in line and any(key in line for key in data.keys()):
                parsed = self.parse_line(line)
                for key, val in parsed:
                    if key in data:
                        data[key].append(val)

        df = pl.DataFrame(data)
        df = df.with_columns(pl.lit(system).alias('system'))
        return df

    def parse_energy_file_OLD(self,
                          file_contents: list[str],
                          data: dict[str, list],
                          system: str) -> pl.DataFrame:
        """
        Parses the contents of an energy calculation using a dictionary of
        energy terms to extract theory-level observables (e.g. EGB vs EPB).

        Arguments:
            file_contents (list[str]): A list of each line from an energy calculation.
            data (dict[str, list]): The relevant energy terms to be scraped from input.
            system (str): The name of the system which will be included as an additional
                kv pair in the returned dataframe. This ensures we can track which portion
                of the calculation we are accounting for (e.g. complex, receptor, ligand).
        Returns:
            (pl.DataFrame): A Polars dataframe of shape (n_frames, n_calculations + system).
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
        """
        Performs a sanity check on the bonded terms which should perfectly cancel out
        (e.g. complex = receptor + ligand). If this is not the case something horrible
        has happened and we can't trust the non-bonded energies either. Additionally
        sets a few terms we will need later such as the number of frames as given by
        the dataframe height and sqrt(n_frames).

        Returns:
            None
        """
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
        """
        Summarizes all processed energy data into a single polars dataframe
        and dumps it to a json file.

        Returns:
            None
        """
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
        """
        For each energy dataframe (GB/PB) compute the ∆G of binding by subtracting out
        relevant contributions in accordance with how this is done under the hood of the
        MMPBSA code.

        Returns:
            None
        """
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
            
            diff_cols.append('∆G Binding')
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
        """
        Ingests a list of Polars dataframes for GB and PB and prints their contents
        in a human-readable form to STDIN. Also saves out the energies to a plain
        text file called `deltaG.txt`.

        Arguments:
            dfs (list[pl.DataFrame]): List of dataframes for GB and PB.

        Returns:
            None
        """
        print_statement = []
        log_statement = []
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
                report = f'{col:<20}{mean:<16.3f}{std:<16.3f}{err:<16.3f}'
                if abs(mean) <= self.tolerance:
                    continue

                if col in ['G gas', '∆G Binding']:
                    print_statement.append('')

                if col == '∆G Binding':
                    log_statement.append(f'{level.strip()}:')
                    log_statement.append(report)

                    if level == 'Poisson Boltzmann':
                        self.free_energy = [mean, std]

                print_statement.append(report)

        print_statement = '\n'.join(print_statement)
        with open(self.path / 'deltaG.txt', 'w') as fout:
            fout.write(print_statement)
        
        if self.log:
            for statement in log_statement:
                logging.info(statement)
        else:
            print(print_statement)

    @staticmethod
    def parse_line(line) -> tuple[list[str], list[float]]:
        """
        Parses a line from mmpbsa_energy to get the various energy terms and values.

        Returns:
            (tuple[list[str], list[float]]): A tuple containing the list of energy
                term names and corresponding energy values.
        """
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
    """
    Performs preprocessing for MM-PBSA runs and manages the pathing to all file
    inputs. Additionally used to write out various cpptraj input files by the
    MMPBSA class.
    """
    def __init__(self,
                 top: Path,
                 traj: Path,
                 path: Path,
                 sels: list[str],
                 first: int,
                 last: int,
                 stride: int,
                 cpptraj_binary: PathLike,
                 n_chunks: int=1):
        self.top = top
        self.traj = traj
        self.path = path
        self.selections = sels
        self.ff = first
        self.lf = last
        self.stride = stride
        self.cpptraj = cpptraj_binary
        self.n_chunks = n_chunks

        self.prepare_topologies()
        self.prepare_trajectories()

        self.trajectory_chunks = {}

        if n_chunks > 1:
            self._count_frames()
            self._split_trajectories()
        else:
            for system, traj in zip(['complex', 'receptor', 'ligand'], self.trajectories):
                self.trajectory_chunks[system] = [traj]

    def prepare_topologies(self) -> None:
        """
        Slices out each sub-topology for the desolvated complex, receptor and
        ligand using cpptraj due to the difficulty of working with AMBER FF
        files otherwise (including PARMED).

        Returns:
            None
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
        subprocess.call(f'{self.cpptraj} -i {script}', shell=True, cwd=str(self.path),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        script.unlink()
        
    def prepare_trajectories(self) -> None:
        """
        Converts DCD trajectory to AMBER CRD format which is explicitly
        required by MM-G(P)BSA.

        Returns:
            None
        """
        self.trajectories = [path.with_suffix('.crd') for path in self.topologies]
        self.pdbs = [path.with_suffix('.pdb') for path in self.topologies]
        
        frame_control = f'start {self.ff}'

        if self.lf > -1:
            frame_control += f' stop {self.lf}'
        
        frame_control += f' offset {self.stride}'
        
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
        subprocess.call(f'{self.cpptraj} -i {name}', shell=True, cwd=str(self.path),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        name.unlink()

    def _count_frames(self) -> None:
        """Count the total number of frames in the processed trajectory."""
        # Use cpptraj to count frames
        count_script = self.path / 'count_frames.in'
        count_out = self.path / 'frame_count.dat'

        script_content = [
            f'parm {self.topologies[0]}',
            f'trajin {self.trajectories[0]}',
            f'trajinfo {self.trajectories[0]} name myinfo',
            'run',
            'quit'
        ]
        self.write_file('\n'.join(script_content), count_script)

        result = subprocess.run(
            f'{self.cpptraj} -i {count_script}',
            shell=True, cwd=str(self.path),
            capture_output=True, text=True
        )

        # Parse frame count from cpptraj output
        for line in result.stdout.split('\n'):
            if 'frames' in line.lower() and 'total' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        self.total_frames = int(part)
                        break

        # Fallback: count lines in trajectory (for ASCII formats) or use file size heuristic
        if not hasattr(self, 'total_frames'):
            # Try to infer from first trajectory file
            self.total_frames = self._estimate_frames()

        count_script.unlink(missing_ok=True)
        logger.debug(f'Total frames in trajectory: {self.total_frames}')

    def _estimate_frames(self) -> int:
        """Estimate frame count by running a quick cpptraj analysis."""
        script = self.path / 'estimate_frames.in'
        script_content = [
            f'parm {self.topologies[0]}',
            f'trajin {self.trajectories[0]}',
            'run',
            'quit'
        ]
        self.write_file('\n'.join(script_content), script)

        result = subprocess.run(
            f'{self.cpptraj} -i {script}',
            shell=True, cwd=str(self.path),
            capture_output=True, text=True
        )

        script.unlink(missing_ok=True)

        # Parse output for frame count
        for line in result.stdout.split('\n'):
            if 'frames' in line.lower():
                for word in line.split():
                    if word.isdigit():
                        return int(word)

        # Default fallback
        return 100

    def _split_trajectories(self) -> None:
        """Split trajectories into chunks for parallel processing."""
        frames_per_chunk = max(1, self.total_frames // self.n_chunks)
        self.trajectory_chunks = {system: [] for system in ['complex', 'receptor', 'ligand']}

        for i, (top, traj, system) in enumerate(zip(
            self.topologies,
            self.trajectories,
            ['complex', 'receptor', 'ligand']
        )):
            for chunk_idx in range(self.n_chunks):
                start_frame = chunk_idx * frames_per_chunk + 1  # cpptraj is 1-indexed

                if chunk_idx == self.n_chunks - 1:
                    # Last chunk gets remaining frames
                    end_frame = self.total_frames
                else:
                    end_frame = (chunk_idx + 1) * frames_per_chunk

                chunk_traj = self.path / f'{system}_chunk{chunk_idx}.crd'

                # Create chunk trajectory
                split_script = self.path / f'split_{system}_{chunk_idx}.in'
                script_content = [
                    f'parm {top}',
                    f'trajin {traj} {start_frame} {end_frame}',
                    f'trajout {chunk_traj} crd',
                    'run',
                    'quit'
                ]
                self.write_file('\n'.join(script_content), split_script)

                subprocess.run(
                    f'{self.cpptraj} -i {split_script}',
                    shell=True, cwd=str(self.path),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

                split_script.unlink()
                self.trajectory_chunks[system].append(chunk_traj)

        logger.debug(f'Split trajectories into {self.n_chunks} chunks of ~{frames_per_chunk} frames each.')

    @property
    def files(self) -> tuple[list[str]]:
        """
        Returns a zip generator containing the output paths, topologies,
        trajectories and pdbs for each system. This is done to ensure we
        have the correct order for housekeeping reasons.

        Returns:
            (tuple[list[str]]): System order, topologies, trajectories and pdbs.
        """
        _order = [self.path / prefix for prefix in ['complex', 'receptor', 'ligand']]
        return zip(_order, self.topologies, self.trajectories, self.pdbs)

    @property
    def files_chunked(self) -> list[tuple]:
        """Returns file info with chunked trajectories for parallel processing."""
        result = []
        for system, top, pdb in zip(
            ['complex', 'receptor', 'ligand'],
            self.topologies,
            self.pdbs
        ):
            prefix = self.path / system
            traj_chunks = self.trajectory_chunks[system]
            result.append((prefix, top, traj_chunks, pdb))

        return result

    @staticmethod
    def write_file(lines: list[str],
                   filepath: PathLike) -> None:
        """
        Given an input of either a list of strings or a single string,
        write input to file. If a list, join by newline characters.

        Arguments:
            lines (list[str]): Input to be written to file.
            filepath (PathLike): Path to the file to be written.

        Returns:
            None
        """
        if isinstance(lines, list):
            lines = '\n'.join(lines)
        with open(str(filepath), 'w') as f:
            f.write(lines)
