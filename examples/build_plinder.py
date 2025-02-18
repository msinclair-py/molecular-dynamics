from glob import glob
from molecular_simulations.build import ImplicitSolvent, PLINDERBuilder
from plinder.core import PlinderSystem
from plinder.core.scores import query_index
from plinder.core.utils.config import get_config

cfg = plinder.core.get_config()

root_path = cfg.data.plinder_dir
base_out_path = '/lus/eagle/projects/FoundEpidem/msinclair/ideals/plinder/systems'

cols_of_interest = ['system_id', 'entry_pdb_id', ]
models = query_index(columns=cols_of_interest)
proteins = []

for protein in proteins:
    # build apo
    cur_out = f'{base_out_path}/{protein}/apo'
    apo = ImplicitBuilder(root_path, protein, cur_out)
    apo.build()

    # build all holo
    ligands = glob(f'{root_path}/{protein}/ligand_files/*.sdf')
    for i, ligand in enumerate(ligands):
        cur_out = f'{base_out_path}/{protein}/lig{i}'
        holo = PLINDERBuilder(root_path, protein, ligand, cur_path)
        holo.build()
