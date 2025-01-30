#!/usr/bin/env python
import gc
from glob import glob
from openmm.app import *
import os
from pdbfixer import PDBFixer
import shutil

class ImplicitSolvent:
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: str, pdb: str, protein: bool=True,
                 rna: bool=False, dna: bool=False):
        self.path = path
        os.makedirs(path, exist_ok=True)

        self.pdb = pdb
        
        ffs = ['leaprc.protein.ff19SB', 
               'leaprc.RNA.Shaw', 
               'leaprc.DNA.OL21']

        self.ffs = [ff for ff, switch in zip(ffs, [protein, rna, dna]) if switch]
    
    def build(self) -> None:
        """
        Orchestrate the various things that need to happen in order
        to produce an explicit solvent system. This includes running
        `pdb4amber`, computing the periodic box size, number of ions
        needed and running `tleap` to make the final system.
        """
        self.pdbfixit()

    def pdbfixit(self) -> str:
        fixer = PDBFixer(filename=self.pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens()
        
        out = f'{self.path}/protein.pdb'
        with open(out, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        del fixer
        gc.collect()
    
class ExplicitSolvent(ImplicitSolvent):
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: str, pdb: str, padding: float=10., protein: bool=True,
                 rna: bool=False, dna: bool=False, polarizable: bool=False):
        super().__init__(path, pdb, protein, rna, dna)
        
        self.out = f'{path}/system'
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
        
class ComplexBuilder(ExplicitSolvent):
    """
    Builds complexes consisting of a biomolecule pdb and small molecule ligand.
    Runs antechamber workflow to generate gaff2 parameters.
    """
    def __init__(self, path: str, pdb: str, lig: str, padding: float=10., **kwargs):
        #from rdkit import Chem
        super().__init__(self, path, pdb, padding, **kwargs)
        
        self.lig = lig
        self.parameterize_ligand()
        
    def parameterize_ligand(self):
        """
        Ensures consistent treatment of all ligand sdf files, generating
        GAFF2 parameters in the form of .frcmod and .lib files. Produces
        a mol2 file for coordinates and connectivity and ensures that
        antechamber did not fail. Hydrogens are added in rdkit which 
        generally does a good job of this.
        """
        fix_resname = f'sed -i s/UNL/LIG/ {self.lig}.pdb'
        cleanse_pdb = f'pdb4amber -i {self.lig}.pdb -o {self.lig}_new.pdb'
        convert_to_gaff = f'antechamber -i {self.lig}_new.pdb -fi pdb -o \
                {self.lig}.mol2 -fo mol2 -at gaff2 -c bcc -s 0 -pf y'
        parmchk2 = f'parmchk2 -i {self.lig}.mol2 -f mol2 -o {self.lig}.frcmod'
        
        tleap_ligand = f"""source leaprc.gaff2
        LIG = loadmol2 {self.lig}.mol2
        loadamberparams {self.lig}.frcmod
        saveoff LIG {self.lig}.lib
        quit
        """
        
        self.add_hydrogens()
        os.system(fix_resname)
        os.system(cleanse_pdb)
        os.system(convert_to_gaff)
        self.move_antechamber_outputs()
        if self.check_sqm():
            os.system(parmchk2)
            leap_file = self.write_leap(tleap_ligand)
            os.system(f'tleap -f {leap_file}')
        else:
            raise RuntimeError('Antechamber has failed! Please take a look at sqm.out')
    
    def add_hydrogens(self) -> None:
        """
        Add hydrogens in rdkit. Atom hybridization is taken from the
        input sdf file and if this is incorrect, hydrogens will be wrong
        too.
        """
        pass
        #mol = Chem.SDMolSupplier(f'{self.lig}.sdf')[0]
        #molH = Chem.AddHs(mol, addCoords=True)
        #Chem.MolToPDBFile(molH, f'{self.lig}.pdb')
        
    def move_antechamber_outputs(self) -> None:
        """
        Remove unneccessary outputs from antechamber. Keep the
        sqm.out file as proof that antechamber did not fail.
        """
        os.remove('sqm.in')
        os.remove('sqm.pdb')
        shutil.move('sqm.out', f'{self.lig}_sqm.out')
        
    def check_sqm(self) -> int:
        """
        Checks for evidence that antechamber calculations exited
        successfully. This is always on the second to last line,
        and if not present, indicates that we failed to produce
        sane parameters for this molecule. In that case, I wish
        you good luck.
        """
        line = open(f'{self.lig}_sqm.out').readlines()[-2]
        if 'Calculation Completed' in line:
            return 1
        return 0
    
    def assemble_system(self, dim, num_ions) -> None:
        """
        Slightly modified from the parent class, now we have to add
        the ligand parameters and assemble a complex rather than just
        placing a biomolecule in the water box.
        """
        tleap_ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        tleap_complex = f"""{tleap_ffs}
        source leaprc.gaff2
        loadamberparams {self.lig}.frcmod
        loadoff {self.lig}.lib
        PROT = loadpdb {self.pdb}
        LIG = loadmol2 {self.lig}.mol2
        
        COMPLEX = combine {{PROT LIG}}
        
        setbox COMPLEX centers
        set COMPLEX box {{{dim} {dim} {dim}}}
        solvatebox COMPLEX {self.water_box} {{0 0 0}}
        
        addions COMPLEX Na+ 0
        addions COMPLEX Cl- 0
        
        addIonsRand COMPLEX Na+ {num_ions} Cl- {num_ions}
        
        saveamberparm COMPLEX {self.out}.prmtop {self.out}.inpcrd
        """
        
        leap_file = self.write_leap(tleap_complex)
        tleap = f'tleap -f {leap_file}'
        os.system(tleap)
        
