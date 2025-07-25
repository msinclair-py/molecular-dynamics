#!/usr/bin/env python
from .build_amber import ImplicitSolvent, ExplicitSolvent
import gc
from glob import glob
import json
from MDAnalysis.lib.util import convert_aa_code
from openbabel import pybel
from openmm import *
from openmm.app import *
import os
from pathlib import Path
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import Sequence
from rdkit import Chem
import shutil
from typing import Dict, List, Union

PathLike = Union[str, Path]
OptPath = Union[str, Path, None]
Sequences = List[Sequence]

class LigandError(Exception):
    """
    Custom Ligand exception to catch parameterization errors.
    """
    def __init__(self, message='This system contains ligands which we cannot model!'):
        self.message = message
        super().__init__(self.message)


class LigandBuilder:
    """
    Parameterizes a ligand molecule and generates all relevant force field files for
    running tleap.
    """
    def __init__(self, path: PathLike, lig: str, lig_number: int=0, file_prefix: str=''):
        self.path = path
        self.lig = path / lig 
        self.ln = lig_number
        self.out_lig = path / f'{file_prefix}{lig.stem}'

    def parameterize_ligand(self) -> None:
        """
        Ensures consistent treatment of all ligand sdf files, generating
        GAFF2 parameters in the form of .frcmod and .lib files. Produces
        a mol2 file for coordinates and connectivity and ensures that
        antechamber did not fail. Hydrogens are added in rdkit which 
        generally does a good job of this.

        Returns:
            None
        """
        ext = self.lig.suffix
        self.lig = self.lig.stem

        convert_to_gaff = f'antechamber -i {self.lig}_prep.mol2 -fi mol2 -o \
                {self.out_lig}.mol2 -fo mol2 -at gaff2 -c bcc -s 0 -pf y -rn LG{self.ln}'
        parmchk2 = f'parmchk2 -i {self.out_lig}.mol2 -f mol2 -o {self.out_lig}.frcmod'
        
        tleap_ligand = f"""source leaprc.gaff2
        LG{self.ln} = loadmol2 {self.out_lig}.mol2
        loadamberparams {self.out_lig}.frcmod
        saveoff LG{self.ln} {self.out_lig}.lib
        quit
        """
        
        if ext == '.sdf':
            self.process_sdf()
        else:
            self.process_pdb()

        self.convert_to_mol2()
        os.system(convert_to_gaff)
        try:
            self.move_antechamber_outputs()
            self.check_sqm()
            os.system(parmchk2)
            leap_file, leap_log = self.write_leap(tleap_ligand)
            os.system(f'tleap -f {leap_file} > {leap_log}')
        except FileNotFoundError:
            raise LigandError(f'Antechamber failed! {self.lig}')
    
    def process_sdf(self) -> None:
        """
        Add hydrogens in rdkit. Atom hybridization is taken from the
        input sdf file and if this is incorrect, hydrogens will be wrong
        too.

        Returns:
            None
        """
        mol = Chem.SDMolSupplier(f'{self.lig}.sdf')[0]
        molH = Chem.AddHs(mol, addCoords=True)
        with Chem.SDWriter(f'{self.lig}_H.sdf') as w:
            w.write(molH)
        
    def process_pdb(self) -> None:
        """
        Ingests a PDB file of a small molecule, adds hydrogens and writes out to
        an SDF file.

        Returns:
            None
        """
        mol = Chem.MolFromPDBFile(f'{self.lig}.pdb')
        molH = Chem.AddHs(mol, addCoords=True)
        with Chem.SDWriter(f'{self.lig}_H.sdf') as w:
            w.write(molH)

    def convert_to_mol2(self) -> None:
        """
        Converts an sdf file to mol2 format using obabel.

        Returns:
            None
        """
        mol = list(pybel.readfile('sdf', f'{self.lig}_H.sdf'))[0]
        mol.write('mol2', f'{self.lig}_prep.mol2', True)

    def move_antechamber_outputs(self) -> None:
        """
        Remove unneccessary outputs from antechamber. Keep the
        sqm.out file as proof that antechamber did not fail.

        Returns:
            None
        """
        os.remove('sqm.in')
        os.remove('sqm.pdb')
        shutil.move('sqm.out', f'{self.lig}_sqm.out')
        
    def check_sqm(self) -> None:
        """
        Checks for evidence that antechamber calculations exited
        successfully. This is always on the second to last line,
        and if not present, indicates that we failed to produce
        sane parameters for this molecule. In that case, I wish
        you good luck.

        Returns:
            None
        """
        line = open(f'{self.lig}_sqm.out').readlines()[-2]

        if 'Calculation Completed' not in line:
            # make sqm.in more tolerant
            # manually run sqm
            # if it still fails then we raise an error
            raise LigandError(f'SQM failed for ligand {self.lig}!')

    def write_leap(self, 
                   inp: str) -> str:
        """
        Writes out a tleap input file and returns the path
        to the file.
        """
        leap_file = f'{self.path}/tleap.in'
        leap_log = f'{self.path}/leap.log'
        with open(leap_file, 'w') as outfile:
            outfile.write(inp)
            
        return leap_file, leap_log


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
        self.ions = None
    
    def build(self) -> None:
        ligs = self.migrate_files()

        if not ligs:
            print(f'No ligands!\n\n{self.pdb}')
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
        os.chdir(self.build_dir) # necessary for antechamber outputs

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
            shutil.copy(str(lig), 
                        str(self.build_dir))

            if self.check_ligand(lig):
                ligands.append(lig.name)


        # handle any potential ions
        if self.ions is not None:
            self.ffs.append('leaprc.water.tip3p')
            self.place_ions()

        return ligands

    def place_ions(self) -> None:
        """
        This is horrible and I apologize profusely if you find yourself
        having to go through the following. Good luck.
        """
        pdb_lines = open(self.pdb).readlines()[:-1]

        if 'END' in pdb_lines[-1]:
            if 'TER' in pdb_lines[-2]:
                ln = -3
            else:
                ln = -2
        elif 'TER' in pdb_lines[-1]:
            ln = -2
        else:
            ln = -1

        try:
            next_atom_num = int(pdb_lines[ln][6:12].strip()) + 1
            next_resid = int(pdb_lines[ln][22:26].strip()) + 1
        except ValueError:
            print(f'ERROR: {self.pdb}')
            raise LigandError
        
        for ion in self.ions:
            for atom in ion:
                # HERE BE DRAGONS
                ion_line = f'ATOM  {next_atom_num:>5}'

                if atom[0].lower() in ['na', 'k', 'cl']:
                    ionname = atom[0] + atom[1]
                    ion_line += f'{ionname:>4}  {ionname:<3} '
                else:
                    ionname = atom[0].upper()
                    ion_line += f'{ionname:>3}   {ionname:<3}'

                coords = ''.join([f'{x:>8.3f}' for x in atom[2:]])
                ion_line += f'{next_resid:>5}    {coords}  0.00  0.00\n'

                pdb_lines.append(ion_line)
                pdb_lines.append('TER\n')

                next_atom_num += 1
                next_resid += 1

        pdb_lines.append('END')
        
        with open(self.pdb, 'w') as f:
            f.write(''.join(pdb_lines))

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
                     chain_map: Dict[str, List[str]]) -> Sequences:
        """
        Checks fasta against actual sequence. Modifies sequence so that 
        it correctly matches in the case of non-canonical residues such
        as phosphorylations (i.e. SER -> SEP).
        """
        fasta = open(self.fasta).readlines()
        remapping = json.load(open(f'{self.path}/chain_mapping.json', 'rb'))
        sequences = []
        for i in range(len(fasta) // 2):
            seq_chain = fasta[2*i].strip()[1:] # strip off > and \n
            chain = remapping[seq_chain]
            one_letter_seq = fasta[2*i+1].strip()
            try:
                three_letter_seq = [convert_aa_code(aa) for aa in one_letter_seq]
            except ValueError:
                print(f'\nUnknown residue in fasta!\n\n{self.pdb}')
                raise LigandError

            try:
                three_letter_seq = self.check_ptms(three_letter_seq,
                                                   chain_map[chain])
                sequences.append(
                    Sequence(chainId=chain, 
                             residues=three_letter_seq)
                )
            except KeyError: # not sure what to do if this fails
                print(f'\nUnknown ligand error!\n\n{self.pdb}')
                raise LigandError

        return sequences

    def check_ptms(self, 
                   sequence: List[str],
                   chain_residues: List[str]) -> List[str]:
        """
        Check the full sequence (from fasta) against the potentially partial
        sequence from the structural model stored in `chain_residues`.
        """
        for residue in chain_residues:
            resID = int(residue.id) - 1 # since 0-indexed in list
            
            try:
                if sequence[resID] != residue.name:
                    sequence[resID] = residue.name
            except IndexError:
                print(f'Sequence length is messed up!\n\n{self.pdb}')
                raise LigandError
        
        return sequence

    def check_ligand(self, ligand: PathLike) -> bool:
        """
        Check ligand for ions and other weird stuff. We need to take care not
        to assume all species containing formal charges are ions, nor that all
        species containing atoms in the cation/anion lists are ions. Good example
        is the multitude of small molecule drugs containing bonded halogens.
        """
        ion = False
        mol = Chem.SDMolSupplier(str(ligand))[0]
        
        ligand = []
        for atom, position in zip(mol.GetAtoms(), mol.GetConformer().GetPositions()):
            symbol = atom.GetSymbol()
            if symbol.lower() in self.cation_list + self.anion_list:
                charge = atom.GetFormalCharge()
                if charge != 0:
                    ion = True
                    sign = '+' if charge > 0 else '-'
                    if abs(charge) > 1:
                        sign = f'{charge}{sign}'

                    ligand.append([symbol, sign] + [x for x in position])

        if ion:
            try:
                self.ions.append(ligand)
            except AttributeError:
                self.ions = [ligand]
            return False

        return True

    @property
    def cation_list(self) -> List[str]:
        return [
            'na', 'k', 'ca', 'mn', 'mg', 'li', 'rb', 'cs', 'cu',
            'ag', 'au', 'ti', 'be', 'sr', 'ba', 'ra', 'v', 'cr',
            'fe', 'co', 'zn', 'ni', 'pd', 'cd', 'sn', 'pt', 'hg',
            'pb', 'al'
        ]

    @property
    def anion_list(self) -> List[str]:
        return [
            'cl', 'br', 'i', 'f'
        ]
  
    
class ComplexBuilder(ExplicitSolvent):
    """
    Builds complexes consisting of a biomolecule pdb and small molecule ligand.
    Runs antechamber workflow to generate gaff2 parameters. Can optionally
    supply precomputed frcmod/lib files by their path + suffix in the 
    lig_param_prefix argument (e.g. /path/to/lig.mol2 or /path/to/lig)
    """
    def __init__(self, path: str, pdb: str, lig: str | list[str], padding: float=10., 
                 lig_param_prefix: str | None = None, **kwargs):
        super().__init__(path, pdb, padding)
        self.lig = Path(lig) if isinstance(lig, str) else [Path(l) for l in lig]
        self.ffs.append('leaprc.gaff2')
        self.build_dir = self.out / 'build'

        if lig_param_prefix is None:
            self.lig_param_prefix = lig_param_prefix
        else:
            prefix = Path(lig_param_prefix)
            self.lig_param_prefix = prefix.parent / prefix.stem

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def build(self) -> None:
        self.build_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(self.build_dir) # necessary for antechamber outputs

        if self.lig_param_prefix is None:
            if isinstance(self.lig, list):
                lig_paths = []
                for i, lig in enumerate(self.lig):
                    lig_paths += self.process_ligand(lig, i)

                self.lig = lig_paths

            else:
                self.lig = self.process_ligand(self.lig)
        else:
            self.lig = self.lig_param_prefix

        if hasattr(self, ion):
            self.add_ion_to_pdb()

        self.prep_pdb()
        self.assemble_system()

    def process_ligand(self, lig: PathLike, prefix: int | None = None) -> PathLike:
        if lig.parent != self.build_dir:
            shutil.copy(lig, self.build_dir)
        
        if prefix is None:
            prefix = ''
            
        lig_builder = LigandBuilder(self.build_dir, lig, file_prefix=prefix)
        lig_builder.parameterize_ligand()

        return lig_builder.lig

    def add_ion_to_pdb(self) -> None:
        ion = [line for line in open(self.ion).readlines() 
               if any(['ATOM' in line, 'HETATM' in line])]
        pdb = [line for line in open(self.pdb).readlines()]
        
        out_pdb = []
        for line in pdb:
            if 'END' in line:
                out_pdb.append(ion)
                out_pdb.append(line)
            else:
                out_pdb.append(line)

        with open(self.pdb, 'w') as f:
            f.write('\n'.join(out_pdb))
        
    def assemble_system(self, dim, num_ions) -> None:
        """
        Slightly modified from the parent class, now we have to add
        the ligand parameters and assemble a complex rather than just
        placing a biomolecule in the water box.
        """
        tleap_ffs = '\n'.join([f'source {ff}' for ff in self.ffs])
        tleap_complex = [
            tleap_ffs,
            'source leaprc.gaff2',
        ]
        
        if not isinstance(self.lig, list):
            self.lig = [self.lig]
        
        LABELS = []
        for i, lig in enumerate(self.lig):
            tleap_complex += [
                f'loadamberparams {lig}.frcmod',
                f'loadoff {lig}.lib',
                f'LG{i} = loadmol2 {lig}.mol2',
            ]

            LABELS.append(f'LG{i}')

        LABELS.append(f'PROT')
        LABELS = ' '.join(LABELS)

        tleap_complex = [
            f'PROT = loadpdb {self.pdb}',
            f'COMPLEX = combine {{LABELS}}',
            'setbox COMPLEX centers',
            f'set COMPLEX box {{{dim} {dim} {dim}}}',
            f'solvatebox COMPLEX {self.water_box} {{0 0 0}}',
            'addions COMPLEX Na+ 0',
            'addions COMPLEX Cl- 0',
            f'addIonsRand COMPLEX Na+ {num_ions} Cl- {num_ions}',
            f'savepdb COMPLEX {self.out}.pdb',
            f'saveamberparm COMPLEX {self.out}.prmtop {self.out}.inpcrd'
        ]
        
        leap_file = self.write_leap('\n'.join(tleap_complex))
        tleap = f'tleap -f {leap_file}'
        os.system(tleap)
        
