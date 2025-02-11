from glob import glob
from molecular_simulations.build import ImplicitSolvent, PLINDERBuilder

root_path = ''
proteins = []
base_out_path = ''

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
