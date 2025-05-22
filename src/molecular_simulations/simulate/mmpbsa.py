from MMPBSA_mods import main
import parmed as pmd
from pathlib import Path
import subprocess
from typing import Union

PathLike = Union[Path, str]

class MMGBSA:
    def __init__(self,
                 top: PathLike,
                 xyz: PathLike,
                 dcd: PathLike,
                 selections: list[str],
                 use_mpi: bool=False,
                 first_frame: int=0,
                 out: str='mmgbsa',
                 **kwargs):
        self.top = Path(top)
        self.xyz = Path(xyz)
        self.traj = Path(dcd)
        self.selections = selections
        self.use_mpi = use_mpi
        self.ff = first_frame
        self.out = out

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self) -> None:
        if not self.traj.with_suffix('.crd').exists():
            self.generate_mdcrd()
        else:
            self.traj = self.traj.with_suffix('.crd')
        
        self.make_topologies()

        self.generate_solvent_input()
        self.calculate()

    def generate_mdcrd(self) -> None:
        """
        Converts DCD trajectory to AMBER CRD format which is explicitly
        required by MM-G(P)BSA.
        """
        cpptraj_in = [
            f'parm {self.top}', 
            f'trajin {self.traj} {self.ff}',
            f'trajout {self.traj.with_suffix(".crd")} crd'
            'run',
            'quit'
        ]

        name = Path('mdcrd.in')
        self.write_script('\n'.join(cpptraj_in), name)
        subprocess.call(f'cpptraj -i {name}', shell=True)

        self.traj = self.traj.with_suffix('.crd')
        name.unlink()

    def generate_solvent_input(self) -> None:
        solv = [
            '&general',
            'startframe=1',
            'endframe=6000',
            'interval=1',
            'verbose=2',
            'keep_files=2',
            '/',
            '&gb',
            'igb=2',
            'saltcon=0.15',
            '/'
        ]

        self.input_file = Path('mmpbsa.in')
        self.write_script('\n'.join(solv), self.input_file)

    def DEPRECATED_make_topologies(self) -> None:
        """
        Generate the relevant topology files needed to run MM-(P)GBSA
        using cpptraj.
        """
        path = self.split_model()
        self.complex = path / 'complex.prmtop'
        self.receptor = path / 'receptor.prmtop'
        self.ligand = path /'ligand.prmtop'



        tleap_in = [
            'source leaprc.protein.ff19SB',
            f'CPX = loadpdb {self.complex.with_suffix(".pdb")}',
            f'REC = loadpdb {self.receptor.with_suffix(".pdb")}',
            f'LIG = loadpdb {self.ligand.with_suffix(".pdb")}',
            f'saveamberparm CPX {self.complex} {self.complex.with_suffix(".inpcrd")}',
            f'saveamberparm REC {self.receptor} {self.receptor.with_suffix(".inpcrd")}',
            f'saveamberparm LIG {self.ligand} {self.ligand.with_suffix(".inpcrd")}',
            'quit'
        ]

        tleap_in = self.write_script('\n'.join(tleap_in))
        subprocess.call(f'tleap -f {tleap_in}', shell=True)
        Path(tleap_in).unlink()

        for prmtop in (self.complex, self.receptor, self.ligand):
            self.unset_IFBOX(prmtop)

    def make_topologies(self) -> PathLike:
        """
        Splits out the input model into the necessary components for 
        further processing by tleap to generate the relevant topologies.

        Returns:
            (PathLike): Path to where the various new prmtops reside.
        """
        path = self.top.parent / 'mmpbsa'
        path.mkdir(exist_ok=True)

        cpptraj_in = [
            f'parm {self.top}',
            'parmstrip :Na+',
            'parmstrip :Cl-',
            'parmstrip :Wat',
            'parmbox nobox',
            f'parmwrite out {self.complex}',
            f'parm {self.complex}',
            f'parmstrip {self.selections[0]}',
            f'parmwrite out {self.receptor}',
            f'parm {self.complex}',
            f'parmstrip {self.selections[1]}',
            f'parmwrite out {self.ligand}'
            'run',
            'quit'
        ]

        script = Path('cpptraj.in')
        self.write_file(cpptraj_in, script)
        subprocess.call(f'cpptraj -i {script}', shell=True)
        script.unlink()
        return path

    def calculate(self) -> None:
        """
        This is nearly a reproduction of the canonical MMPBSA.py script.
        Here we pipe in the correct pieces ourselves thus bypassing the need
        for the CLI altogether (which is both clunky and allows us to remain
        completely within the python API).
        """
        if self.use_mpi:
            from mpi4py import MPI
        else:
            from MMPBSA_mods.fake_mpi import MPI
        
        main.setup_run()
        app = main.MMPBSA_App(MPI)
        
        self.bypass_cli()
        app.get_cl_args(self.args)

        app.read_input_file()
        app.process_input()
        app.check_for_bad_input()
        app.loadcheck_prmtops()
        app.file_setup()

        app.run_mmpbsa()

        app.parse_output_files()
        app.write_final_outputs()
        app.finalize()

    def bypass_cli(self,) -> None:
        """
        """
        self.args = [
            '-O',
            '-i', str(self.input_file),
            '-o', str(self.out),
            '-cp', str(self.complex),
            '-rp', str(self.receptor),
            '-lp', str(self.ligand),
            '-y', str(self.traj)
        ]

    def unset_IFBOX(self, 
                    prmtop_file: PathLike) -> None:
        """AMBER sucks"""
        lines = [line for line in open(str(prmtop_file)).readlines()]
        lines[8] = lines[8][:56] + f'0:>8' + lines[8][64:]

        self.write_prmtop(lines, prmtop_file)

    @staticmethod
    def write_file(lines: list[str],
                   filepath: PathLike) -> None:
        with open(str(filepath), 'w') as f:
            f.write(lines)

class MMPBSA(MMGBSA):
    def __init__(self,
                 top: PathLike,
                 dcd: PathLike,
                 use_mpi: bool=False,
                 first_frame: int=0,
                 last_frame: int=-1,
                 out: str='mmpbsa',
                 **kwargs):
        super().__init__(top, dcd, use_mpi, first_frame, last_frame, out)

    def bypass_cli(self,) -> None:
        """
        """
        self.args = [
            '-O',
            '-i', str(self.input_file),
            '-o', str(self.out),
            '-sp', str(self.top),
            '-cp', str(self.complex),
            '-rp', str(self.receptor),
            '-lp', str(self.ligand),
            '-y', str(self.traj)
        ]
    
    def generate_solvent_input(self) -> None:
        solv = [
            '&general',
            'startframe=1',
            'endframe=6000',
            'interval=1',
            'verbose=2',
            'keep_files=2',
            '/',
            '&gb',
            'igb=2',
            'saltcon=0.15',
            '/',
            '&pb',
            'istrng=0.15',
        ]

        self.input_file = Path('mmpbsa.in')
        self.write_script('\n'.join(solv), self.input_file)
