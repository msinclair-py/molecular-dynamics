from molecular_simulations.build import ExplicitSolvent
from pathlib import Path

pdb = Path('/path/to/input.pdb')
out_path = Path('/path/to/simulation/inputs')

builder = ExplicitSolvent(out_path, pdb)
builder.build()
