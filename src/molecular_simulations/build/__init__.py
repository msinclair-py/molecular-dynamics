from Bio.PDB import MMCIFParser, PDBIO
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
