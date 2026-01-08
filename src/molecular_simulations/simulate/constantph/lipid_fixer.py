import MDAnalysis as mda
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

class LipidFixer:
    def __init__(self, prmtop: Path, inpcrd: Path, out: Path, ff_xml: Path = None):
        print("Loading structure with MDAnalysis...")
        self.u = mda.Universe(str(prmtop), str(inpcrd))
        self.out = out
        self.ff_xml = ff_xml

        # Mappings
        self.lipid_components = {'PC', 'PA', 'PE', 'OL', 'GL'}
        self.lipid_mappings = {
            frozenset(['PA', 'PE', 'OL']): 'POPE',
            frozenset(['PA', 'PC', 'OL']): 'POPC',
        }
        
        self.lipid_names = {'POPE', 'POPC', 'DOPC', 'DOPE', 'DPPC'}
        self.protein_names = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYX', 'GLN', 'GLU', 
                         'GLY', 'HIS', 'HID', 'HIE', 'HIP', 'ILE', 'LEU', 'LYS', 
                         'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                         'ACE', 'NME', 'NHE'}
        
        self.templates = {}
        if ff_xml:
            self.load_templates(ff_xml)
        
        print(f"Loaded {self.u.atoms.n_atoms} atoms, {self.u.atoms.n_residues} residues")
    
    def load_templates(self, ff_xml):
        tree = ET.parse(ff_xml)
        root = tree.getroot()
        
        type_to_element = {}
        for atom_type in root.findall('.//AtomTypes/Type'):
            type_name = atom_type.get('name')
            element = atom_type.get('element')
            if type_name and element:
                type_to_element[type_name] = element
        
        for residue in root.findall('.//Residue'):
            res_name = residue.get('name')
            if res_name in self.lipid_names:
                atoms = {}
                bonds = []
                
                for atom in residue.findall('Atom'):
                    atom_name = atom.get('name')
                    atom_type = atom.get('type')
                    element = type_to_element.get(atom_type, 'C')
                    atoms[atom_name] = {'type': atom_type, 'element': element}
                
                for bond in residue.findall('Bond'):
                    atom1 = bond.get('atomName1')
                    atom2 = bond.get('atomName2')
                    bonds.append((atom1, atom2))
                
                self.templates[res_name] = {
                    'atoms': atoms, 
                    'bonds': bonds, 
                    'order': list(atoms.keys())
                }
                print(f"  Loaded template for {res_name}: {len(atoms)} atoms")
    
    def merge_lipids(self):
        """Merge lipid residues - build new atom/residue lists"""
        print("\nMerging lipids...")
        
        residues = list(self.u.residues)
        
        # Build new residue and atom lists
        self.output_residues = []
        self.output_atoms = []  # List of (atom_object, new_resid, new_resname)
        
        i = 0
        new_resid = 1
        merged_count = 0
        
        while i < len(residues):
            if i + 2 < len(residues):
                res1, res2, res3 = residues[i], residues[i+1], residues[i+2]
                
                # Check if lipid pattern
                if (res1.resname in self.lipid_components and 
                    res2.resname in self.lipid_components and 
                    res3.resname in self.lipid_components):
                    
                    pattern = frozenset([res1.resname, res2.resname, res3.resname])
                    
                    if pattern in self.lipid_mappings:
                        lipid_name = self.lipid_mappings[pattern]
                        
                        # Get all atoms from 3 residues
                        all_atoms = res1.atoms + res2.atoms + res3.atoms
                        
                        # Just rename to match template (NO REORDERING!)
                        if lipid_name in self.templates:
                            self.rename_atoms_only(all_atoms, lipid_name)
                        
                        # Add atoms in their ORIGINAL order (preserves geometry)
                        for atom in all_atoms:
                            self.output_atoms.append((atom, new_resid, lipid_name))
                        
                        self.output_residues.append({
                            'resid': new_resid,
                            'resname': lipid_name,
                            'chain': 'M',
                            'n_atoms': len(all_atoms)
                        })
                        
                        new_resid += 1
                        merged_count += 1
                        i += 3
                        continue
            
            # Not a lipid pattern, keep as is
            res = residues[i]
            for atom in res.atoms:
                self.output_atoms.append((atom, new_resid, res.resname))
            
            self.output_residues.append({
                'resid': new_resid,
                'resname': res.resname,
                'chain': res.segid if hasattr(res, 'segid') else '',
                'n_atoms': len(res.atoms)
            })
            
            new_resid += 1
            i += 1
        
        print(f"Merged {merged_count} lipids into {new_resid-1} total residues")
        print(f"Total output atoms: {len(self.output_atoms)}")
    
    def rename_atoms_only(self, atoms, lipid_name):
        """Rename atoms to match template WITHOUT reordering (preserves geometry)"""
        template = self.templates[lipid_name]
        
        # Get atom types if available
        if hasattr(atoms, 'types'):
            types = atoms.types
        else:
            types = [None] * len(atoms)
        
        # Build type->template name mapping
        type_to_names = defaultdict(list)
        for tpl_name, info in template['atoms'].items():
            type_to_names[info['type']].append(tpl_name)
        
        # Match atoms to template names
        used_names = set()
        new_names = []
        
        for i, atom in enumerate(atoms):
            atom_type = types[i] if types[i] else None
            
            if atom_type:
                candidates = [n for n in type_to_names.get(atom_type, []) if n not in used_names]
                if candidates:
                    new_name = candidates[0]
                    new_names.append(new_name)
                    used_names.add(new_name)
                else:
                    new_names.append(atom.name)
            else:
                new_names.append(atom.name)
        
        # Rename atoms IN PLACE (no reordering!)
        atoms.names = new_names
    
    def rename_and_reorder_atoms(self, atoms, lipid_name):
        """Rename and reorder atoms to match template"""
        template = self.templates[lipid_name]
        template_order = template['order']
        
        # Get atom types if available
        if hasattr(atoms, 'types'):
            types = atoms.types
        else:
            # Fallback: guess from names
            types = [None] * len(atoms)
        
        # Build type->template name mapping
        type_to_names = defaultdict(list)
        for tpl_name, info in template['atoms'].items():
            type_to_names[info['type']].append(tpl_name)
        
        # Match atoms to template names
        used_names = set()
        name_map = {}  # atom_index -> new_name
        
        for i, atom in enumerate(atoms):
            atom_type = types[i] if types[i] else None
            
            if atom_type:
                candidates = [n for n in type_to_names.get(atom_type, []) if n not in used_names]
                if candidates:
                    new_name = candidates[0]
                    name_map[i] = new_name
                    used_names.add(new_name)
        
        # Rename atoms
        new_names = []
        for i, atom in enumerate(atoms):
            if i in name_map:
                new_names.append(name_map[i])
            else:
                new_names.append(atom.name)
        
        atoms.names = new_names
        
        # Reorder to match template
        name_to_atom = {atom.name: atom for atom in atoms}
        reordered_atoms = []
        
        for template_name in template_order:
            if template_name in name_to_atom:
                reordered_atoms.append(name_to_atom[template_name])
        
        # Add unmatched
        for atom in atoms:
            if atom not in reordered_atoms:
                reordered_atoms.append(atom)
        
        # Return as AtomGroup
        return mda.AtomGroup([atom for atom in reordered_atoms])
    
    def assign_chains(self):
        """Assign chain IDs to output residues"""
        print("\nAssigning chain IDs...")
        
        for res in self.output_residues:
            resname = res['resname']
            if resname in self.lipid_names:
                res['chain'] = 'M'
            elif resname in self.protein_names:
                res['chain'] = 'P'
            elif resname == 'LIG':
                res['chain'] = 'L'
            elif resname in {'Na+', 'NA', 'SOD'}:
                res['chain'] = 'N'
            elif resname in {'Cl-', 'CL', 'CLA'}:
                res['chain'] = 'C'
            else:
                res['chain'] = 'W'
    
    def write_pdb(self, conect_resnames):
        """Write PDB directly from output data structures"""
        print(f"\nWriting PDB to {self.out}...")
        
        # Extract ALL data into plain arrays upfront
        print("  Extracting atom data...")
        n_atoms = len(self.output_atoms)
        
        atom_names = []
        resnames = []
        resids = []
        chains = []
        elements = []
        old_indices = []  # Track original indices for later
        
        resid_to_info = {res['resid']: res for res in self.output_residues}
        
        # CRITICAL: We need to extract coordinates in the REORDERED sequence
        # The atoms in output_atoms have been reordered, but their .index still points
        # to the ORIGINAL position. We need to map the reordered sequence to coordinates.
        
        for atom, resid, resname in self.output_atoms:
            res_info = resid_to_info[resid]
            
            atom_names.append(atom.name)
            resnames.append(resname)
            resids.append(resid)
            chains.append(res_info['chain'])
            elements.append(atom.element[:2] if hasattr(atom, 'element') and atom.element else '  ')
            # Store original index - we'll use this to get the right coordinates
            old_indices.append(atom.index)
        
        # Get coordinates in the reordered sequence by indexing with old_indices
        all_coords = self.u.atoms.positions
        coords = all_coords[old_indices]  # This gets coordinates in output atom order
        
        print(f"  Writing {n_atoms} atoms...")
        with open(self.out, 'w', buffering=8192*1024) as f:
            # CRYST1
            if self.u.dimensions is not None:
                box = self.u.dimensions
                a, b, c = box[0], box[1], box[2]
                alpha, beta, gamma = box[3], box[4], box[5]
                f.write(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n")
            
            # Build old->new serial mapping
            old_to_new_serial = {old_indices[i]: i+1 for i in range(n_atoms)}
            
            # Write atoms
            for i in range(n_atoms):
                atom_serial = i + 1
                x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
                
                record = "HETATM" if resnames[i] not in self.protein_names else "ATOM  "
                
                if atom_serial < 100000:
                    f.write(f"{record}{atom_serial:5d} {atom_names[i]:<4s} {resnames[i]:<4s}{chains[i]}{resids[i]:<8}"
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elements[i]:>2s}  \n")
                else:
                    f.write(f"{record}{atom_serial:6d}{atom_names[i]:<4s} {resnames[i]:<4s}{chains[i]}{resids[i]:<8}"
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elements[i]:>2s}  \n")
            
            print(f"  Wrote {n_atoms} atoms")
            
            # Don't write CONECT records - atoms are in original order, not template order
            # OpenMM will infer bonds from geometry
            print(f"  Skipping CONECT records (OpenMM will infer bonds from geometry)")
            
            f.write("END\n")
            
            print(f"Done! Wrote {self.out}")
    
    def build(self):
        self.merge_lipids()
        self.assign_chains()
        self.write_pdb({'POPE', 'POPC', 'DOPC', 'DOPE', 'LIG'})


# Usage:
# fixer = LipidFixer(
#     prmtop=Path('system.prmtop'),
#     inpcrd=Path('system.inpcrd'),
#     out=Path('system_fixed.pdb'),
#     ff_xml=Path('lipid17.xml')
# )
# fixer.build()
