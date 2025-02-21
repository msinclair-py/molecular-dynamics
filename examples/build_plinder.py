import os
from molecular_simulations.build.build_amber import PLINDERBuilder, LigandError
from pathlib import Path
from plinder.core import PlinderSystem
from plinder.core.scores import query_index
import plinder.core.utils.config

cfg = plinder.core.get_config()
cfg.data.plinder_dir = '' # set this to where to place downloads
root_simulation_dir = Path('') # where to build systems

columns_of_interest = [
    'system_id', 'entry_pdb_id', 'entry_oligomeric_state', 'entry_resolution',
    'system_num_ligand_chains', 'system_protein_chains_num_unresolved_residues',
    'ligand_is_artifact', 'system_protein_chains_validation_unknown_residue_count',
    'system_ligand_validation_unknown_residue_count', 'system_num_covalent_ligands',
]

filters = [
    ('system_num_ligand_chains', '>=', 1),
    ('ligand_is_artifact', '==', False),
    ('system_ligand_validation_unknown_residue_count', '==', 0.),
    ('system_num_covalent_ligands', '==', 0),
]

df = query_index(columns=columns_of_interest, filters=filters)
df.drop_duplicates(subset='system_id', inplace=True)

for system_id in df['system_id']:
    path = f'{cfg.data.plinder_dir}/{system_id}'
    out_path = root_simulation_dir / system_id
    # check if we have already downloaded this system
    if not os.path.exists(path):
        # spin up system
        plinder_system = PlinderSystem(system_id=system_id)
    
        # download ligand sdf files
        plinder_system.ligand_sdfs

        # download receptor pdb file
        plinder_system.receptor_pdb
        
        builder = PLINDERBuilder(path, 
                                 system_id=system_id, 
                                 out=out_path)
        try:
            builder.build()
        except LigandError:
            continue
