from Bio.PDB import MMCIFParser, PDBIO
import MDAnalysis as mda
from pathlib import Path
from typing import Union

from .build_amber import ExplicitSolvent, ImplicitSolvent

PathLike = Union[Path, str]

def convert_cif_to_pdb(cif: PathLike) -> PathLike:
    """
    Helper function to convert a cif file to a pdb file using biopython.
    """
    if not isinstance(cif, Path):
        cif = Path(cif)
    pdb = cif.with_suffix('.pdb')
    
    parser = MMCIFParser()
    structure = parser.get_structure('protein', str(cif))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb))

    return pdb

def add_chains(pdb: PathLike,
               first_res: int=1,
               last_res: int=-1) -> PathLike:
    """
    Helper function to add chain IDs to a model.
    """
    u = mda.Universe(pdb)
    u.add_TopologyAttr('chainID')

    if last_res == -1:
        last_res = len(u.n_residues)

    chain_A = u.select_atoms(f'resid {first_res} to {last_res}')
    chain_A.atoms.chainIDs = 'A'

    if last_res != -1:
        final_res = len(u.n_residues)

        chain_B = u.select_atoms(f'resid {last_res} to {final_res}')
        chain_B.atoms.chainIDs = 'B'

    with mda.Writer(Path(pdb).with_suffix('_withchains.pdb')) as W:
        W.write(u.atoms)
