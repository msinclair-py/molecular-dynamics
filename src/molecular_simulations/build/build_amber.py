#!/usr/bin/env python
import gc
from glob import glob
from MDAnalysis.lib.util import convert_aa_code
from openmm.app import *
import os
from pathlib import Path
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import Sequence
import shutil
from typing import Dict, List, Union

PathLike = Union[str, Path]
OptPath = Union[str, Path, None]
Sequences = List[Sequence]

class LigandError(Exception):
    def __init__(self, message='This system contains ligands which we cannot model!'):
        self.message = message
        super.__init__(self.message)

class ImplicitSolvent:
    """
    Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    """
    def __init__(self, path: PathLike, pdb: str, out: OptPath=None, 
                 protein: bool=True, rna: bool=False, dna: bool=False,
                 phos_protein: bool=False, mod_protein: bool=False):
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
        
class PLINDERBuilder(ImplicitSolvent):
    """
    Builds complexes consisting of a biomolecule pdb and small molecule ligand.
    Runs antechamber workflow to generate gaff2 parameters.
    """
    def __init__(self, 
                 path: PathLike, 
                 system_id: str,
                 out: PathLike, 
                 **kwargs):
        super().__init__(path / system_id, 'receptor.pdb', out / system_id, 
                         protein=True, rna=True, dna=True, phos_protein=True,
                         mod_protein=True, **kwargs)
        self.system_id = system_id
        self.ffs.append('leaprc.gaff2')
        self.build_dir = self.out / 'build'
    
    def build(self) -> None:
        ligs = self.migrate_files()

        if not ligs:
            raise LigandError

        self.ligs = self.ligand_handler(ligs)
        self.assemble_system()

    def ligand_handler(self, ligs: List[PathLike]) -> List[PathLike]:
        ligands = []
        for i, lig in enumerate(ligs):
            lig_builder = LigandBuilder(self.build_dir, lig, i)
            lig_builder.parameterize_ligand()
            ligands.append(os.path.basename(lig)[:-4])

        return ligands
        
    def migrate_files(self) -> List[str]:
        os.makedirs(str(self.build_dir), exist_ok=True)

        # grab the sequence file to complete protein modeling
        shutil.copy(str(self.path / 'sequences.fasta'),
                    str(self.build_dir))
        self.fasta = str(self.build_dir / 'sequences.fasta')
        
        # fix and move pdb
        self.prep_protein()

        # move ligand(s)
        ligands = []
        lig_files = self.path / 'ligand_files'
        ligs = [Path(lig) for lig in glob(str(lig_files) + '/*.sdf')]
        for lig in ligs:
            if self.check_ligand(lig):
                shutil.copy(str(lig), 
                            str(self.build_dir))
                ligands.append(lig.name)

        return ligands

    def assemble_system(self) -> None:
        """
        Slightly modified from the parent class, now we have to add
        the ligand parameters and assemble a complex rather than just
        placing a biomolecule in the water box.
        """
        tleap_complex = [f'source {ff}' for ff in self.ffs]
        structs = [f'PROT = loadpdb {self.pdb}']
        combine = 'COMPLEX = combine{PROT'
        for i, lig in enumerate(self.ligs):
            ligand = self.build_dir / lig
            tleap_complex += [f'loadamberparams {ligand}.frcmod', 
                              f'loadoff {ligand}.lib']
            structs += [f'LG{i} = loadmol2 {ligand}.mol2']
            combine += f' LG{i}'
        
        combine += '}'
        tleap_complex += structs
        tleap_complex.append(combine)
        tleap_complex += [
            'set default PBRadii mbondi3',
            f'savepdb COMPLEX {self.out}/system.pdb',
            f'saveamberparm COMPLEX {self.out}/system.prmtop {self.out}/system.inpcrd',
            'quit'
        ]

        tleap_complex = '\n'.join(tleap_complex)
        leap_file = self.build_dir / 'tleap.in'
        with open(str(leap_file), 'w') as outfile:
            outfile.write(tleap_complex)
            
        tleap = f'tleap -f {leap_file}'
        os.system(tleap)

    def prep_protein(self) -> None:
        raw_pdb = self.path / self.pdb
        prep_pdb = self.build_dir / 'prepped.pdb'
        self.pdb = self.build_dir / 'protein.pdb'
       
        # complex workflow for modeling missing residues
        self.triage_pdb(raw_pdb, prep_pdb)
        
        # remove hydrogens (-y) and waters (-d) from the input PDB
        pdb4amber = f'pdb4amber -i {prep_pdb} -o {self.pdb} -y -d'
        os.system(pdb4amber)

    def triage_pdb(self, broken_pdb: PathLike, 
                   repaired_pdb: PathLike) -> str:
        """
        Runs PDBFixer to repair missing loops and ensure structure is
        in good shape. Runs a check against the sequence provided by
        PLINDER and ensures that any non-canonical residues are represented
        in the sequence properly.
        """
        fixer = PDBFixer(filename=str(broken_pdb))
        chains = [chain for chain in fixer.topology.chains()]
        chain_map = {chain.id: [res for res in chain.residues()] 
                     for chain in chains}
        # non-databank models do not have SEQRES and therefore no
        # sequence data to model missing residues
        fixer.sequences = self.inject_fasta(chain_map)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        
        with open(str(repaired_pdb), 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

    def inject_fasta(self, 
                     chain_map: Dict[str, List[topology.Residue]]) -> Sequences:
        """
        Checks fasta against actual sequence. Modifies sequence so that 
        it correctly matches in the case of non-canonical residues such
        as phosphorylations (i.e. SER -> SEP).
        """
        fasta = open(self.fasta).readlines()
        sequences = []
        for i in range(len(fasta) // 2):
            chain = fasta[2*i].strip().split('.')[-1]
            one_letter_seq = fasta[2*i+1].strip()
            three_letter_seq = [convert_aa_code(aa) for aa in one_letter_seq]
            try:
                three_letter_seq = self.check_ptms(three_letter_seq,
                                                   chain_map[chain])
                sequences.append(
                    Sequence(chainId=chain, 
                             residues=three_letter_seq)
                )
            except KeyError: # not sure what to do if this fails
                break

        return sequences

    def check_ptms(self, 
                   sequence: List[str],
                   chain_residues: List[topology.Residue]) -> List[str]:
        """
        Check the full sequence (from fasta) against the potentially partial
        sequence from the structural model stored in `chain_residues`.
        """
        for residue in chain_residues:
            resID = int(residue.id) - 1 # since 0-indexed in list
            
            if sequence[resID] != residue.name:
                sequence[resID] = residue.name
        
        return sequence

    @staticmethod
    def check_ligand(ligand: PathLike) -> bool:
        """
        Check ligand for ions and other weird stuff.
        """
        #ion_list = ['na', 'k', 'ca', 'mn', 'mg', 'li', 'rb', 'cs', 'cu',
        #            'ag', 'au', 'ti', 'be', 'sr', 'ba', 'ra', 'v', 'cr',
        #            'fe', 'co', 'zn', 'ni', 'pd', 'cd', 'sn', 'pt', 'hg',
        #            'pb', 'al']
        #lig_lines = [line for line in open(str(ligand)).readlines()]

        #for line in lig_lines:
        #    resname = line[12:16].strip()
        #    if any([ion in resname.lower() for ion in ion_list]):
        #        return False

        #    segid = line[76:].strip()
        #    if '-' in segid or '+' in segid:
        #        return False
        from rdkit import Chem
        mol = Chem.SDMolSupplier(str(ligand))[0]
        if mol.GetNumAtoms() < 5: # covers larger cations/anions
            return False

        return True
  
class LigandBuilder:
    def __init__(self, path: PathLike, lig: str, lig_number: int=0):
        self.path = path
        self.lig = path / lig 
        self.ln = lig_number

    def parameterize_ligand(self) -> None:
        """
        Ensures consistent treatment of all ligand sdf files, generating
        GAFF2 parameters in the form of .frcmod and .lib files. Produces
        a mol2 file for coordinates and connectivity and ensures that
        antechamber did not fail. Hydrogens are added in rdkit which 
        generally does a good job of this.
        """
        if self.lig.name[-4:] == '.sdf':
            self.lig = str(self.lig)[:-4]

        fix_resname = f'sed -i "" "s/UNL/LG{self.ln}/" {self.lig}.pdb'
        cleanse_pdb = f'pdb4amber -i {self.lig}.pdb -o {self.lig}_new.pdb'
        convert_to_gaff = f'antechamber -i {self.lig}_new.pdb -fi pdb -o \
                {self.lig}.mol2 -fo mol2 -at gaff2 -c bcc -s 0 -pf y'
        parmchk2 = f'parmchk2 -i {self.lig}.mol2 -f mol2 -o {self.lig}.frcmod'
        
        tleap_ligand = f"""source leaprc.gaff2
        LG{self.ln} = loadmol2 {self.lig}.mol2
        loadamberparams {self.lig}.frcmod
        saveoff LG{self.ln} {self.lig}.lib
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
        from rdkit import Chem
        mol = Chem.SDMolSupplier(f'{self.lig}.sdf')[0]
        molH = Chem.AddHs(mol, addCoords=True)
        Chem.MolToPDBFile(molH, f'{self.lig}.pdb')
        
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

    def write_leap(self, inp: str) -> str:
        """
        Writes out a tleap input file and returns the path
        to the file.
        """
        leap_file = f'{self.path}/tleap.in'
        with open(leap_file, 'w') as outfile:
            outfile.write(inp)
            
        return leap_file
    

class ComplexBuilder(ExplicitSolvent):
    """
    Builds complexes consisting of a biomolecule pdb and small molecule ligand.
    Runs antechamber workflow to generate gaff2 parameters.
    """
    def __init__(self, path: str, pdb: str, lig: str, padding: float=10., **kwargs):
        from rdkit import Chem
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
        mol = Chem.SDMolSupplier(f'{self.lig}.sdf')[0]
        molH = Chem.AddHs(mol, addCoords=True)
        Chem.MolToPDBFile(molH, f'{self.lig}.pdb')
        
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
        
