import MDAnalysis as mda
import string

def assign_chainids(u: mda.Universe,
                    terminus_selection: str='name OXT') -> mda.Universe:
    if not hasattr(u.atoms, 'chainIDs'):
        u.add_TopologyAttr('chainIDs')

    termini_atoms = u.select_atoms(terminus_selection)
    termini_resindices = set(termini_atoms.resindices)

    def get_chain_label(index):
        if index < 26:
            return string.ascii_uppercase[index]
        first = string.ascii_uppercase[(index // 26) - 1]
        second = string.ascii_uppercase[index % 26]
        return first + second

    chain_index = 0
    for residue in u.residues:
        residue.atoms.chainIDs = get_chain_label(chain_index)

        if residue.resindex in termini_resindices:
            chain_index += 1

    return u
