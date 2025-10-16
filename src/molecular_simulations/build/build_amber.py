#!/usr/bin/env python
from openmm.app import PDBFile
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Union

PathLike = Union[str, Path]
OptPath = Union[str, Path, None]

class ImplicitSolvent:
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: OptPath, pdb: str, protein: bool=True,
                 rna: bool=False, dna: bool=False, phos_protein: bool=False,
                 mod_protein: bool=False, out: OptPath=None,
                 amberhome: str='', **kwargs):
        if path is None:
            self.path = Path(pdb).parent
        elif isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path

        self.path = self.path.resolve()
        self.path.mkdir(exist_ok=True, parents=True)

        self.pdb = pdb

        if out is not None:
            self.out = self.path / out
        else:
            self.out = self.path / 'system.pdb' 

        self.out = self.out.resolve()

        self.amberhome = amberhome

        switches = [protein, rna, dna, phos_protein, mod_protein]
        ffs = [
            'leaprc.protein.ff19SB', 
            'leaprc.RNA.Shaw', 
            'leaprc.DNA.OL21',
            'leaprc.phosaa19SB',
            'leaprc.protein.ff14SB_modAA'
        ]
        
        self.ffs = [
            ff for ff, switch in zip(ffs, switches) if switch
        ]
        
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        if not hasattr(self, 'tleap'):
            self.tleap = f'{self.amberhome}tleap'

    def build(self) -> None:
        """
        Orchestrate the various things that need to happen in order
        to produce an explicit solvent system. This includes running
        `pdb4amber`, computing the periodic box size, number of ions
        needed and running `tleap` to make the final system.

        Returns:
            None
        """
        self.tleap_it()

    def tleap_it(self) -> None:
        """
        While more painful, tleap is more stable for system building.
        Runs the input PDB through tleap with the FF19SB protein forcefield
        and whichever other forcefields were turned on.

        Returns:
            None
        """
        ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        tleap_in = f"""
        {ffs}
        prot = loadpdb {self.pdb}
        set default pbradii mbondi3
        savepdb prot {self.out}
        saveamberparm prot {self.out.with_suffix('.prmtop')} {self.out.with_suffix('.inpcrd')}
        quit
        """

        self.temp_tleap(tleap_in)
    
    def write_leap(self, 
                   inp: str) -> Path:
        """
        Writes out a tleap input file and returns the path
        to the file.

        Returns:
            (Path): Path to tleap input file.
        """
        leap_file = f'{self.path}/tleap.in'
        with open(leap_file, 'w') as outfile:
            outfile.write(inp)
            
        return Path(leap_file)

    def temp_tleap(self,
                   inp: str) -> None:
        """
        Writes a temporary file for tleap and then runs tleap. This
        makes handling parallel tleap runs much simpler as we are avoiding
        the different workers overwriting tleap inputs and getting confused.

        Arguments:
            inp (str): The tleap input file contents as a single string.

        Returns:
            None
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.in') as temp_file:
            temp_file.write(inp)
            temp_file.flush()
            tleap_command = f'{self.tleap} -f {temp_file.name}'
            print(tleap_command)
            os.system(tleap_command)
    
class ExplicitSolvent(ImplicitSolvent):
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: PathLike, pdb: PathLike, padding: float=10., protein: bool=True,
                 rna: bool=False, dna: bool=False, phos_protein: bool=False,
                 mod_protein: bool=False, polarizable: bool=False, amberhome: str='', **kwargs):
        super().__init__(path, pdb, protein, rna, dna, phos_protein, 
                         mod_protein, None, amberhome, **kwargs)
        
        self.out = self.path / 'system'
        self.pad = padding
        self.ffs.extend(['leaprc.water.opc'])
        self.water_box = 'OPCBOX'
        
        if polarizable:
            self.ffs[0] = 'leaprc.protein.ff15ipq'
            self.ffs[-1] = 'leaprc.water.spceb'
            self.water_box = 'SPCBOX'
    
    def build(self) -> None:
        """
        Orchestrate the various things that need to happen in order
        to produce an explicit solvent system. This includes running
        `pdb4amber`, computing the periodic box size, number of ions
        needed and running `tleap` to make the final system.

        Returns:
            None
        """
        self.prep_pdb()
        dim = self.get_pdb_extent()
        num_ions = self.get_ion_numbers(dim**3)
        self.assemble_system(dim, num_ions)
        self.clean_up_directory()

    def prep_pdb(self) -> None:
        """
        Runs input PDB through `pdb4amber` to ensure it is compliant enough
        that tleap won't freak out on us later. Removes any explicit hydrogens
        from the input structure to avoid name mismatches.

        Returns:
            None
        """
        os.system(f'{self.amberhome}pdb4amber -i {self.pdb} -o {self.path}/protein.pdb -y')
        self.pdb = f'{self.path}/protein.pdb'
        
    def assemble_system(self, dim: float, num_ions: int) -> None:
        """
        Build system in tleap.

        Arguments:
            dim (float): Longest dimension in Angstrom.
            num_ions (int): Explicit number of ions to achieve 150mM.

        Returns:
            None
        """
        tleap_ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        tleap_complex = f"""{tleap_ffs}
        PROT = loadpdb {self.pdb}
        
        setbox PROT centers
        set PROT box {{{dim} {dim} {dim}}}
        solvatebox PROT {self.water_box} {{0 0 0}}
        
        addions PROT Na+ 0
        addions PROT Cl- 0
        
        addIonsRand PROT Na+ {num_ions} Cl- {num_ions}
        
        savepdb PROT {self.out}.pdb
        saveamberparm PROT {self.out}.prmtop {self.out}.inpcrd
        quit
        """
        
        leap_file = self.write_leap(tleap_complex)
        tleap = f'{self.amberhome}tleap -f {leap_file}'
        os.system(tleap) 

    def get_pdb_extent(self) -> int:
        """
        Identifies the longest axis of the protein in terms of X/Y/Z
        projection. Not super accurate but likely good enough for determining
        PBC box size. Returns longest axis length + 2 times the padding
        to account for +/- padding.

        Returns:
            (int): Longest dimension with 2 times padding in Angstrom.
        """
        lines = [line for line in open(self.pdb).readlines() if 'ATOM' in line]
        xs, ys, zs = [], [], []
        
        for line in lines:
            xs.append(float(line[30:38].strip()))
            ys.append(float(line[38:46].strip()))
            zs.append(float(line[46:54].strip()))
        
        xtent = (max(xs) - min(xs))
        ytent = (max(ys) - min(ys))
        ztent = (max(zs) - min(zs))
        
        return int(max([xtent, ytent, ztent]) + 2 * self.pad)
    
    def clean_up_directory(self) -> None:
        """
        Remove leap log. This is placed wherever the script calling it
        runs and likely will throw errors if multiple systems are
        being iteratively built.

        Returns:
            None
        """
        os.remove('leap.log')
        (self.path / 'build').mkdir(exist_ok=True)
        for f in self.path.glob('*'):
            if not any([ext in f.name for ext in ['.prmtop', '.inpcrd', 'build']]):
                f.rename(f.parent / 'build' / f.name)
        
    @staticmethod
    def get_ion_numbers(volume: float) -> int:
        """
        Returns the number of Chloride? ions required to achieve 150mM
        concentration for a given volume. The number of Sodium counter
        ions should be equivalent.

        Arguments:
            volume (float): Volume of box in cubic Angstrom.

        Returns:
            (int): Number of ions for 150mM NaCl.
        """
        return round(volume * 10e-6 * 9.03)
