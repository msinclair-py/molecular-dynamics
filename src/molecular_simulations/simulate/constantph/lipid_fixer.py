import MDAnalysis as mda
import numpy as np
from openmm.app import AmberPrmtopFile, AmberInpcrdFile, Modeller
from openmm import app
from openmm.app import element
from pathlib import Path

def merge_lipid21_residues(topology):
    """
    Merge Lipid21 3-residue lipids (PA/PE/OL) into single POPE residues.
    Returns a new topology with merged residues.
    """
    # Create a new topology
    new_top = app.Topology()
    
    # Copy box vectors
    new_top.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    
    # Map old atoms to new atoms for bond copying
    atom_map = {}
    
    # Get list of residues to process
    residues = list(topology.residues())
    i = 0
    
    while i < len(residues):
        residue = residues[i]
        
        # Check if this is a lipid triad (PA, PE, OL)
        if (residue.name == 'PA' and 
            i + 2 < len(residues) and 
            residues[i+1].name == 'PE' and 
            residues[i+2].name == 'OL'):
            
            # This is a POPE lipid - merge the three residues
            pa_res = residues[i]
            pe_res = residues[i+1]
            ol_res = residues[i+2]
            
            # Get or create chain
            chain_id = pa_res.chain.id
            new_chain = None
            for c in new_top.chains():
                if c.id == chain_id:
                    new_chain = c
                    break
            if new_chain is None:
                new_chain = new_top.addChain(chain_id)
            
            # Create single POPE residue (use PE residue number as the ID)
            new_residue = new_top.addResidue('POPE', new_chain, str(pe_res.id))
            
            # Add all atoms from PA, PE, OL to the new POPE residue
            for res in [pa_res, pe_res, ol_res]:
                for atom in res.atoms():
                    new_atom = new_top.addAtom(
                        atom.name, 
                        atom.element, 
                        new_residue
                    )
                    atom_map[atom] = new_atom
            
            # Skip the next two residues since we processed them
            i += 3
            
        else:
            # Not a lipid - copy residue as-is
            chain_id = residue.chain.id
            new_chain = None
            for c in new_top.chains():
                if c.id == chain_id:
                    new_chain = c
                    break
            if new_chain is None:
                new_chain = new_top.addChain(chain_id)
            
            new_residue = new_top.addResidue(residue.name, new_chain, str(residue.id))
            
            for atom in residue.atoms():
                new_atom = new_top.addAtom(
                    atom.name, 
                    atom.element, 
                    new_residue
                )
                atom_map[atom] = new_atom
            
            i += 1
    
    # Copy bonds
    for bond in topology.bonds():
        atom1, atom2 = bond
        new_top.addBond(
            atom_map[atom1], 
            atom_map[atom2],
            type=bond.type,
            order=bond.order
        )
    
    return new_top

def generate_exclusions(out):
    titratable = ' '.join([
        'ASP', 'ASH', 
        'CYS', 'CYX',
        'GLU', 'GLH', 
        'HIS', 'HSE', 'HSD', 'HSP'
        'LYS', 'LYN', 
    ])

    u = mda.Universe(str(out))
    phosphates = u.select_atoms('name P31')
    zmin = np.min(phosphates.positions, axis=0)[-1]
    zmax = np.max(phosphates.positions, axis=0)[-1]

    if np.abs(zmin - zmax) > 50.: # OpenMM split membrane across PBC
        zmin, zmax = zmax, zmin

    sel = u.select_atoms('resname {titratable} and (prop z > {zmin} or prop z < {zmax})')
    
    # check termini since the amber capping precludes us from titrating
    # them with openmm
    termini = []
    protein = u.select_atoms('protein').residues
    first = protein[0]
    last = protein[-1]

    if first.resname == 'ACE':
        first = protein[1]
    
    for terminus in [first, last]:
        if terminus.resname in titratable:
            termini.append(terminus.resid)
    
    if sel:
        resids = [str(x) for x in sel.residues.resids]
        if termini:
            resids += [str(x) for x in termini]

        with open(out.parent / 'exclusions.txt', 'w') as fout:
            fout.write(''.join(resids))

if __name__ == '__main__':
    path = Path('systems/wt')
    out = path / 'system_merged.pdb'

    # Load your files
    prmtop = AmberPrmtopFile(str(path / 'system.prmtop'))
    inpcrd = AmberInpcrdFile(str(path / 'system.inpcrd'))
    
    # Merge the lipid residues
    merged_topology = merge_lipid21_residues(prmtop.topology)
    
    with open(out, 'w') as fout:
        app.PDBFile.writeFile(merged_topology, inpcrd.positions, fout)

    # Get protonation exclusion list
    generate_exclusions(out)
