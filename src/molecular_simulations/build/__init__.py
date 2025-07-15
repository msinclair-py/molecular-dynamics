from pathlib import Path
from typing import Union

from .build_amber import ExplicitSolvent, ImplicitSolvent
from .build_interface import InterfaceBuilder

try:
    from .build_ligand import LigandBuilder, PLINDERBuilder, ComplexBuilder
except ImportError: # no rdkit in environment
    pass

PathLike = Union[Path, str]

def convert_cif_with_biopython(cif: PathLike) -> PathLike:
    """
    Helper function to convert a cif file to a pdb file using biopython.
    """
    from Bio.PDB import MMCIFParser, PDBIO

    if not isinstance(cif, Path):
        cif = Path(cif)
    pdb = cif.with_suffix('.pdb')
    
    parser = MMCIFParser()
    structure = parser.get_structure('protein', str(cif))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb))

    return pdb

def convert_cif_with_gemmi(cif: PathLike) -> PathLike:
    import gemmi
    structure = gemmi.read_structure(str(cif))
    structure.write(str(cif.with_suffix('.pdb')))

def add_chains(pdb: PathLike,
               first_res: int=1,
               last_res: int=-1) -> PathLike:
    """
    Helper function to add chain IDs to a model.
    """
    import MDAnalysis as mda

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
