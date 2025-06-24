from copy import deepcopy
from itertools import permutations
import numpy as np
from numpy import vectorize
from pathlib import Path
from typing import Any, Dict, Union

ChainDict = Dict[str, Dict[str, Union[float, set, np.ndarray]]]
PathLike = Union[Path, str]


class ipSAE:
    def __init__(self,
                 pdb_file: PathLike,
                 plddt_file: PathLike,
                 pae_file: PathLike):
        self.parser = ModelParser(pdb_file)
        self.plddt_file = plddt_file
        self.pae_file = pae_file

        self.preprocess()

    def preprocess(self) -> None:
        self.parser.parse_structure_file()
        self.CA_atoms = np.array([res['atom_num'] - 1 
                                  for res in self.parser.residues])
        self.CB_atoms = np.array([res['atom_num'] - 1 
                                  for res in self.parser.cb_residues])
        self.coordinates = np.array([res['coor'] 
                                     for res in self.parser.cb_residues])
        self.chains = np.array(self.parser.chains)
        self.token_array = np.array(self.parser.token_mask)
        self.residue_types = np.array([res['res'] 
                                       for res in self.parser.residues])

        self.unique_chains = np.unique(self.chains)
        self.init_chain_dicts()

        self.scorer = ScoreCalculator(self.chains, 
                                      self.parser.chain_types,
                                      deepcopy(self.chain_dict_set), 
                                      deepcopy(self.chain_dict_zeros),
                                      self.parser.residues) 

    def run(self) -> None:
        distances = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        distances = np.sqrt((distances ** 2).sum(axis=2))
        plddt = self.load_plddt_file()
        pae = self.load_pae_file()
        iptm = self.get_interchain_iptm()
        
        self.scorer.compute_scores(distances, plddt, pae)

    def load_plddt_file(self) -> np.ndarray:
        data = np.load(self.plddt_file)
        plddt_arr = np.array(data['plddt'] * 100.)
        plddt = plddt_arr[np.ix_(self.token_array.astype(bool))]

        return plddt

    def load_pae_file(self) -> np.ndarray:
        data = np.load(self.pae_file)
        pae = np.array(data['pae'])[np.ix_(self.token_array.astype(bool), 
                                           self.token_array.astype(bool))]

        return pae

    def get_interchain_iptm(self) -> ChainDict:
        summary_file = str(self.pae_file.with_suffix('.json')).replace('pae', 'confidence')
        iptm = deepcopy(self.chain_dict)

        data_summary = json.load(open(summary_file, 'r'))
        chain_pair_iptm = data_summary['pair_chains_iptm']
        
        chain_zero = ord('A')
        for chain1, chain2 in permutations(zip(self.unique_chains, self.unique_chains)):
            nchain1 = ord(chain1) - chain_zero
            nchain2 = ord(chain2) - chain_zero
            iptm[chain1][chain2] = chain_per_iptm[str(nchain1)][str(nchain2)]

        return iptm

    def init_chain_dicts(self) -> None:
        self.chain_dict_zeros = {chain1: {chain2: 0 
                                          for chain2 in self.unique_chains 
                                          if chain1 != chain2} 
                                 for chain1 in self.unique_chains}
        
        self.chain_dict_set = {chain1: {chain2: set()
                                        for chain2 in self.unique_chains 
                                        if chain1 != chain2} 
                                 for chain1 in self.unique_chains}
   

class ScoreCalculator:
    def __init__(self,
                 chains: np.ndarray,
                 chain_pair_type: Dict[str, str],
                 dict_set: ChainDict,
                 dict_float: ChainDict,
                 residues: Any,
                 pdockq_cutoff: float=8.,
                 pae_cutoff: float=12.):
        self.chains = chains
        self.unique_chains = np.unique(chains)
        self.chain_pair_type = chain_pair_type
        self.dict_set = dict_set
        self.dict_float = dict_float
        self.residues = residues
        self.n_res = len(self.residues)
        self.pDockQ_cutoff = pdockq_cutoff
        self.PAE_cutoff = pae_cutoff
        
    def compute_scores(self,
                       distances: np.ndarray,
                       plddt: np.ndarray,
                       pae: np.ndarray) -> None:
        self.distances = distances
        self.plddt = plddt
        self.pae = pae

        self.compute_pDockQ_scores()
        self.compute_LIS()
        self.compute_ipTM_ipSAE()

    def compute_pDockQ_scores(self) -> None:
        self.pDockQ = deepcopy(self.dict_float)
        self.pDockQ2 = deepcopy(self.dict_float)
        
        for chain1, chain2 in permutations(zip(self.unique_chains, self.unique_chains)):
            n_pairs = 0
            _sum = 0.
            for i in range(self.n_res):
                if self.chains[i] != chain1:
                    continue

                valid_pairs = (self.chains == chain2) & (self.distances[i] <= self.pDockQ_cutoff)
                n_pairs += np.sum(valid_pairs)
                if valid_pairs.any():
                    self.unique[chain1][chain2].add(i)
                    chain2_residues = np.where(valid_pairs)[0]
                    pae_list = self.pae[i][valid_pairs]
                    pae_list_ptm = self.compute_ptm(pae_list, 10.)
                    _sum += pae_list_ptm.sum()

                    for residue in chain2_residues:
                        self.unique[chain1][chain2].add(residue)

            if n_pairs > 0:
                residues = list(self.unique[chain1][chain2])
                n_res = len(residues)
                mean_plddt = self.plddt[residues].mean()
                x = mean_plddt * np.log10(n_pairs)
                self.pDockQ[chain1][chain2] = self.pDockQ_score(x)
                
                mean_ptm = _sum / n_pairs
                x = mean_plddt * mean_ptm
                self.pDockQ2[chain1][chain2] = self.pDockQ2_score(x)

    def compute_LIS(self) -> None:
        self.LIS = deepcopy(self.dict_float)

        for chain1, chain2 in permutations(zip(self.unique_chains, self.unique_chains)):
            mask = (self.chains[:, None] == chain1) & (self.chains[None, :] == chain2)
            selected_pae = self.pae[mask]

            if selected_pae.size > 0:
                valid_pae = selected_pae[selected_pae <= 12]
                if valid_pae.size > 0:
                    scores = (12 - valid_pae) / 12
                    avg_score = np.mean(scores)
                    self.LIS[chain1][chain2] = avg_score

    def compute_ipTM_ipSAE(self) -> None:
        n0_chain = deepcopy(self.dict_float)
        d0_chain = deepcopy(self.dict_float)
        dist_valid_pairs_counts = deepcopy(self.dict_float)
        unique_residues_chain1 = deepcopy(self.dict_set)
        unique_residues_chain2 = deepcopy(self.dict_set)
        iptm_d0_chain_byres = self.initialize_dict_np(self.n_res)
        ipsae_d0_chain_byres = self.initialize_dict_np(self.n_res)

        for chain1, chain2 in permutatations(zip(self.unique_chains, self.unique_chains)):
            L = np.sum(chains == chain1) + np.sum(chains == chain2)
            n0_chain[chain1][chain2] = L
            d0_chain[chain1][chain2] = self.calc_d0(L, self.chain_pair_type[chain1][chain2])

            n_res = len(list(self.unique[chain1][chain2]))
            ptm_matrix = np.zeros((self.n_res, self.n_res))
            ptm_matrix = self.compute_ptm()

            valid_pairs_iptm = (self.chains == chain2)
            valid_pairs_matrix = (self.chains == chain2) & (self.pae < self.pae_cutoff)

            for i in range(self.n_res):
                if chains[i] != chain1:
                    continue

                valid_pairs_ipsae = valid_pairs_matrix[i]
                if valid_pairs_iptm.any():
                    iptm_d0_chain_byres[chain1][chain2][i] = ptm_matrix[i, valid_pairs_iptm].mean()
                if valid_pairs_ipsae.any():
                    ipsae_d0_chain_byres[chain1][chain2][i] = ptm_matrix[i, valid_pairs_ipsae].mean()

                valid_pair_counts[chain1][chain2] += np.sum(valid_pairs_ipsae)
                if valid_pairs_ipsae.any():
                    resnum_i = self.residues[i]['resnum']
                    unique_residues_chain1[chain1][chain2].add(resnum_i)
                    for j in np.where(valid_pairs_ipsae)[0]:
                        resnum_j = self.residues[j]['resnum']
                        unique_residues_chain2[chain1][chain2].add(resnum_j)

                valid_pairs = (chains == chain2) & (self.pae[i] < self.pae_cutoff) & (self.distances[i] < self.distance_cutoff)
                dist_valid_pairs_counts[chain1][chain2] += np.sum(valid_pairs)

                if valid_pairs.any():
                    resnum_i = self.residues[i]['resnum']
                    dist_unique_residues_chain1[chain1][chain2].add(resnum_i)
                    for j in np.where(valid_pairs)[0]:
                        resnum_j = residues[j]['resnum']
                        dist_unique_residues_chain2[chain1][chain2].add(resnum_j)

    def initialize_dict_np(self,
                           size: int) -> ChainDict:
        return {chain1: {chain2: np.zeros((size)) for chain2 in vals.keys()} 
                for chain1, vals in self.dict_float.items()}

    @staticmethod
    def pDockQ_score(x) -> float:
        return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

    @staticmethod
    def pDockQ2_score(x) -> float:
        return 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005
    
    @vectorize
    @staticmethod
    def compute_ptm(x: float, 
                    d0: float) -> float:
        return 1. / (1 + (x / d0)**2)
        
    @staticmethod
    def compute_d0(L: int, 
                   pair_type: str) -> float:
        L = np.array(L, dtype=float)
        L = np.maximum(27, L)
        min_value = 1.

        if pair_type == 'nucleic_acid':
            min_value = 2.

        return np.maximum(min_value, 1.24 * (L - 15) ** (1/3) - 1.8)

class ModelParser:
    def __init__(self,
                 pdb: PathLike):
        self.pdb = pdb

        self.token_mask = []
        self.residues = []
        self.chains = []

    def parse_structure_file(self) -> None:
        if self.pdb.suffix == 'pdb':
            line_parser = self.parse_pdb_line
        else:
            line_parser = self.parse_cif_line
        
        field_num = 0
        lines = open(self.pdb).readlines()
        for line in lines:
            if line.startswith('_atom_site.'):
                _, field_name = line.strip().split('.')
                fields[field_name] = field_num
                field_num += 1

            if any([line.startswith(atom) for atom in ['ATOM', 'HETATM']]):
                atom = line_parser(line, fields)
                
                self.token_mask.append(0)
                if atom['atom_name'] == 'CA' or 'C1' in atom['atom_name']:
                    self.token_mask[-1] = 1
                    self.residues.append(atom)
                    self.chains.append(atom['chain_id'])

                elif atom['atom_name'] == 'CB' or 'C3' in atom['atom_name']:
                    self.cb_residues.append(atom)
                
                elif atom['res'] == 'GLY' and atom['atom_name'] == 'CA':
                    self.cb_residues.append(atom)

    def classify_chains(self) -> None:
        chains = np.unique(self.chains)
        self.chain_types = {chain: 'protein' for chain in chains}
        for chain in chains:
            indices = np.where(chains == chain)[0]
            chain_residues = self.residue_types[indices]
            if any([r in chain_residues for r in self.nucleic_acids]):
                self.chain_types[chain] = 'nucleic_acid'

    @property
    def nucleic_acids(self) -> list[str]:
        return ['DA', 'DC', 'DT', 'DG', 'A', 'C', 'U', 'G']
    
    @staticmethod
    def parse_pdb_line(line: str, **kwargs) -> dict[str, Any]:
        atom_num = line[6:11].strip()
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_id = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()

        return ModelParser.package_line(atom_num, atom_name, residue_name, 
                                        chain_id, residue_id, x, y, z)

    @staticmethod
    def parse_cif_line(line: str,
                       fields: dict[str, int]) -> dict[str, Any]:
        _split = line.split()
        atom_num = _split[fields['id']]
        atom_name = _split[fields['label_atom_id']]
        residue_name = _split[fields['label_comp_id']]
        chain_id = _split[fields['label_asym_id']]
        residue_id = _split[fields['label_seq_id']]
        x = _split[fields['Cartn_x']]
        y = _split[fields['Cartn_y']]
        z = _split[fields['Cartn_z']]

        if residue_id == '.':
            return None

        return ModelParser.package_line(atom_num, atom_name, residue_name, 
                                        chain_id, residue_id, x, y, z)
    
    @staticmethod
    def package_line(atom_num: str, 
                     atom_name: str, 
                     residue_name: str, 
                     chain_id: str, 
                     residue_id: str, 
                     x: str, 
                     y: str, 
                     z: str) -> dict[str, Any]:
        return {
            'atom_num': int(atom_num),
            'atom_name': atom_name,
            'coor': np.array([float(i) for i in [x, y, z]]),
            'res': residue_name,
            'chainid': chain_id,
            'resnum': int(residue_id),
        }
