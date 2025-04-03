#!/usr/bin/env python
import gc
from glob import glob
import os
from pathlib import Path
from pdbfixer import PDBFixer
import shutil
from typing import Dict, List, Union

PathLike = Union[str, Path]
OptPath = Union[str, Path, None]

class ImplicitSolvent:
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: PathLike, pdb: str, protein: bool=True,
                 rna: bool=False, dna: bool=False, phos_protein: bool=False,
                 mod_protein: bool=False, out: OptPath=None):
        self.path = path
        os.makedirs(str(path), exist_ok=True)

        self.pdb = pdb

        if out is not None:
            self.out = out
        else:
            self.out = path

        os.makedirs(str(self.out), exist_ok=True)

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
    
    def build(self) -> None:
        """
        Orchestrate the various things that need to happen in order
        to produce an explicit solvent system. This includes running
        `pdb4amber`, computing the periodic box size, number of ions
        needed and running `tleap` to make the final system.
        """
        self.pdbfixit()

    def pdbfixit(self, add_hydrogen=True) -> str:
        fixer = PDBFixer(filename=str(self.path / self.pdb))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens()
        
        out = self.out / 'protein.pdb'
        with open(str(out), 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        del fixer
        gc.collect()
    
class ExplicitSolvent(ImplicitSolvent):
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: PathLike, pdb: PathLike, padding: float=10., protein: bool=True,
                 rna: bool=False, dna: bool=False, polarizable: bool=False):
        super().__init__(path, pdb, protein, rna, dna)
        
        self.out = Path(path) / 'system'
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
        """
        self.prep_pdb()
        dim = self.get_pdb_extent()
        num_ions = self.get_ion_numbers(dim**3)
        self.assemble_system(dim, num_ions)
        self.clean_up_directory()

    def prep_pdb(self):
        os.system(f'pdb4amber -i {self.pdb} -o {self.path}/protein.pdb -y')
        self.pdb = f'{self.path}/protein.pdb'
        
    def assemble_system(self, dim: float, num_ions: int) -> None:
        """
        Build system in tleap.
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
        tleap = f'tleap -f {leap_file}'
        os.system(tleap)
        
    def write_leap(self, inp: str) -> str:
        """
        Writes out a tleap input file and returns the path
        to the file.
        """
        leap_file = f'{self.path}/tleap.in'
        with open(leap_file, 'w') as outfile:
            outfile.write(inp)
            
        return leap_file

    def get_pdb_extent(self) -> int:
        """
        Identifies the longest axis of the protein in terms of X/Y/Z
        projection. Not super accurate but likely good enough for determining
        PBC box size. Returns longest axis length + 2 times the padding
        to account for +/- padding.
        """
        lines = [line for line in open(self.pdb).readlines() if 'ATOM' in line]
        xs, ys, zs = [], [], []
        
        for line in lines:
            x, y, z = [float(i.strip()) for i in line[26:54].split()]
            xs.append(x)
            ys.append(y)
            zs.append(z)
        
        xtent = (max(xs) - min(xs))
        ytent = (max(ys) - min(ys))
        ztent = (max(zs) - min(zs))
        
        return int(max([xtent, ytent, ztent]) + 2 * self.pad)
    
    def clean_up_directory(self) -> None:
        """
        Remove leap log. This is placed wherever the script calling it
        runs and likely will throw errors if multiple systems are
        being iteratively built.
        """
        os.remove('leap.log')
        os.makedirs(f'{self.path}/build', exist_ok=True)
        for f in glob(f'{self.path}/*'):
            if not any([ext in os.path.basename(f) for ext in ['.prmtop', '.inpcrd', 'build']]):
                path, name = os.path.split(f)
                shutil.move(f, f'{path}/build/{name}')
        
    @staticmethod
    def get_ion_numbers(volume: int) -> float:
        """
        Returns the number of Chloride? ions required to achieve 150mM
        concentration for a given volume. The number of Sodium counter
        ions should be equivalent.
        """
        return round(volume * 10e-6 * 9.03)
