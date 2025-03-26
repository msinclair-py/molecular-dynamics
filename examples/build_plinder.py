from molecular_simulations import AuroraSettings
from parsl import get_config, python_app
from pathlib import Path
from plinder.core.scores import query_index

input_dir = Path('/lus/flare/projects/FoundEpidem/plinder/2024-06/v2/systems') 
root_simulation_dir = Path('/lus/flare/projects/FoundEpidem/plinder/simulations')

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

@python_app
def construct(system_id, input_path, output_path):
    molecular_simulations.build.build_amber import LigandError, PLINDERBuilder
    import os
    from pathlib import Path
    
    path = path / system_id
    out_path = root_simulation_dir / system_id
    # check if we have already downloaded this system
    if not os.path.exists(path):
        builder = PLINDERBuilder(input_path, 
                                 system_id=system_id, 
                                 out=output_path)
        try:
            builder.build()
        except LigandError:
            return system_id

future = []
for system_id in df['system_id']:
    futures.append(construct(system_id, input_path, output_path))

bad_systems = [x.result() for x in futures]

with open('bad_systems.txt') as out:
    for system in bad_systems:
        out.write(f'{system}\n')
