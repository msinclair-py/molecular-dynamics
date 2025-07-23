from itertools import permutations
import numpy as np
from numpy import vectorize
from pathlib import Path
import polars as pl
from typing import Any, Union

PathLike = Union[Path, str]
OptPath = Union[Path, str, None]

class ipSAE:
    """
    Compute the interaction prediction Score from Aligned Errors for a model.
    Adapted from https://doi.org/10.1101/2025.02.10.637595. Currently supports
    only outputs which provide plddt and pae data which limits us to Boltz and
    AlphaFold.

    Arguments:
        structure_file (PathLike): Path to PDB/CIF model.
        plddt_file (PathLike): Path to plddt npy file.
        pae_file (PathLike): Path to pae npy file.
        out_path (PathLike | None): Defaults to None. Path for outputs, or if None,
            will use the parent path from the plddt file.
    """
    def __init__(self, 
                 structure_file: PathLike,
                 plddt_file: PathLike,
                 pae_file: PathLike,
                 out_path: OptPath=None):
        self.parser = ModelParser(structure_file)
        self.plddt_file = Path(plddt_file)
        self.pae_file = Path(pae_file)

        self.path = Path(out_path) if out_path is not None else self.plddt_file.parent
        self.path.mkdir(exist_ok=True)

    def parse_structure_file(self) -> None:
        """
        Runs parser to read in structure file and extract relevant details.

        Returns:
            None
        """
        self.parser.parse_structure_file()
        self.parser.classify_chains()
        self.coordinates = np.vstack([res['coor'] for res in self.parser.residues])
        self.token_array = np.array(self.parser.token_mask, dtype=bool)

    def prepare_scorer(self) -> None:
        """
        Prepares scorer for computing various scores.

        Returns:
            None
        """
        chains = np.array(self.parser.chains)
        chain_types = self.parser.chain_types
        residue_types = np.array([res['res'] for res in self.parser.residues])

        self.scorer = ScoreCalculator(chains=chains,
                                      chain_pair_type=chain_types,
                                      residues=residue_types) 

    def run(self) -> None:
        """
        Main logic of class. Parses structure file, computes distogram, unpacks
        pLDDT and PAE, feeds data to scorer and saves out scores.

        Returns:
            None
        """
        self.parse_structure_file()

        distances = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        distances = np.sqrt((distances ** 2).sum(axis=2))
        pLDDT = self.load_pLDDT_file()
        PAE = self.load_PAE_file()

        self.prepare_scorer()
        self.scorer.compute_scores(distances, pLDDT, PAE)

        self.scores = self.scorer.scores
        self.save_scores()

    def save_scores(self) -> None:
        """
        Saves scores dataframe to a parquet file.

        Returns:
            None
        """
        self.scores.write_parquet(self.path / 'ipSAE_scores.parquet')

    def load_pLDDT_file(self) -> np.ndarray:
        """
        Loads pLDDT file and scales data by 100.

        Returns:
            (np.ndarray): Scaled pLDDT array.
        """
        data = np.load(str(self.plddt_file))
        pLDDT_arr = np.array(data['plddt'] * 100.)

        return pLDDT_arr

    def load_PAE_file(self) -> np.ndarray:
        """
        Loads PAE file and returns data.

        Returns:
            (np.ndarray): Array of PAE values.
        """
        data = np.load(str(self.pae_file))['pae']
        return data

class ScoreCalculator:
    """
    Computes various model quality scores including: pDockQ, pDockQ2, LIS, ipTM and
    the ipSAE score.

    Arguments:
        chains (np.ndarray): Array of chainIDs.
        chain_pair_type (dict[str, str]): Dictionary mapping of chainID to chain type.
        n_residues (int): Number of residues total in structure.
        pdockq_cutoff (float): Defaults to 8.0 Å.
        pae_cutoff (float): Defaults to 12.0 Å.
        dist_cutoff (float): Defaults to 10.0 Å.
    """
    def __init__(self,
                 chains: np.ndarray,
                 chain_pair_type: dict[str, str],
                 n_residues: int,
                 pdockq_cutoff: float=8.,
                 pae_cutoff: float=12.,
                 dist_cutoff: float=10.):
        self.chains = chains
        self.unique_chains = np.unique(chains)
        self.chain_pair_type = chain_pair_type
        self.n_res = n_residues
        self.pDockQ_cutoff = pdockq_cutoff
        self.PAE_cutoff = pae_cutoff
        self.dist_cutoff = dist_cutoff

        self.permute_chains()

    def compute_scores(self,
                       distances: np.ndarray,
                       pLDDT: np.ndarray,
                       PAE: np.ndarray) -> None:
        """
        Based on the input distance, pLDDT and PAE matrices, compute the pairwise pDockQ, pDockQ2,
        LIS, ipTM and ipSAE scores.

        Returns:
            None
        """
        self.distances = distances
        self.pLDDT = pLDDT
        self.PAE = PAE

        results = []
        for chain1, chain2 in self.permuted:
            pDockQ, pDockQ2 = self.compute_pDockQ_scores(chain1, chain2)
            LIS = self.compute_LIS(chain1, chain2)
            ipTM, ipSAE = self.compute_ipTM_ipSAE(chain1, chain2)

            results.append([chain1, chain2, pDockQ, pDockQ2, LIS, ipTM, ipSAE])

        self.df = pl.DataFrame(np.array(results), schema={'chain1': str, 
                                                          'chain2': str, 
                                                          'pDockQ': float, 
                                                          'pDockQ2': float,
                                                          'LIS': float,
                                                          'ipTM': float,
                                                          'ipSAE': float})
        self.get_max_values()

    def compute_pDockQ_scores(self,
                              chain1: str,
                              chain2: str) -> tuple[float, float]:
        """
        Computes both the pDockQ and pDockQ2 scores for the interface between two chains.
        pDockQ is dependent solely on the pLDDT matrix while pDockQ2 is dependent on both
        pLDDT and the PAE matrix.

        Arguments:
            chain1 (str): The string name of the first chain.
            chain2 (str): The string name of the first chain.

        Returns:
            (tuple[float, float]): A tuple of the pDockQ and pDockQ2 scores respectively.
        """
        n_pairs = 0
        _sum = 0.
        residues = set()
        for i in range(self.n_res):
            if self.chains[i] == chain1:
                continue

            valid_pairs = (self.chains == chain2) & (self.distances[i] <= self.pDockQ_cutoff)
            n_pairs += np.sum(valid_pairs)
            if valid_pairs.any():
                residues.add(i)
                chain2_residues = np.where(valid_pairs)[0]
                pae_list = self.PAE[i][valid_pairs]
                pae_list_ptm = self.compute_pTM(pae_list, 10.)
                _sum += pae_list_ptm.sum()

                for residue in chain2_residues:
                    residues.add(residue)

        if n_pairs > 0:
            residues = list(residues)
            n_res = len(residues)
            mean_pLDDT = self.pLDDT[residues].mean()
            x = mean_pLDDT * np.log10(n_pairs)
            pDockQ = self.pDockQ_score(x)

            mean_pTM = _sum / n_pairs
            x = mean_pLDDT * mean_pTM
            pDockQ2 = self.pDockQ2_score(x)

        return pDockQ, pDockQ2

    def compute_LIS(self,
                    chain1: str, 
                    chain2: str) -> float:
        """
        Computes Local Interaction Score (LIS) which is based on a subset of the 
        predicted aligned error using a cutoff of 12. Values range in the interval 
        (0, 1] and can be interpreted as how accurate a fold is within the error 
        cutoff where a mean error of 0 yields a LIS value of 1 and a mean error
        that approaches 12 has a LIS value that approaches 0.
        Adapted from: https://doi.org/10.1101/2024.02.19.580970.

        Arguments:
            chain1 (str): The string name of the first chain.
            chain2 (str): The string name of the second chain.
        Returns:
            (float): The LIS value for both chains.
        """
        mask = (self.chains[:, None] == chain1) & (self.chains[None, :] == chain2)
        selected_pae = self.PAE[mask]

        LIS = 0.
        if selected_pae.size:
            valid_pae = selected_pae[selected_pae < 12]
            if valid_pae.size:
                scores = (12 - valid_pae) / 12
                avg_score = np.mean(scores)
                LIS = avg_score

        return LIS

    def compute_ipTM_ipSAE(self,
                           chain1: str,
                           chain2: str) -> tuple[float, float]:
        """
        Computes the ipTM and ipSAE scores for a given pair of chains. These operations
        are combined since they rely on very similar processing of the data.

        Arguments:
            chain1 (str): The first chain to compare.
            chain2 (str): The second chain to compare.

        Returns:
            (tuple[float]): A tuple containing the ipTM and ipSAE scores respectively.
        """
        pair_type = 'protein'
        if 'nucleic' in [self.chain_pair_type[chain1], self.chain_pair_type[chain2]]:
            pair_type = 'nucleic'

        L = np.sum(self.chains == chain1) + np.sum(self.chains == chain2)
        d0_chain = self.compute_d0(L, pair_type)

        pTM_matrix_chain = self.compute_pTM(self.PAE, d0_chain)
        ipTM_byres = np.zeros((pTM_matrix_chain.shape[0]))

        valid_pairs_ipTM = (self.chains == chain2)
        ipTM_byres = np.array([0.])
        if valid_pairs_ipTM.any():
            ipTM_byres = np.mean(pTM_matrix_chain[:, valid_pairs_ipTM], axis=0)

        valid_pairs_matrix = (self.chains == chain2) & (self.PAE < self.PAE_cutoff)
        valid_pairs_ipSAE = valid_pairs_matrix

        ipSAE_byres = np.array([0.])
        if valid_pairs_ipSAE.any():
            ipSAE_byres = np.mean(pTM_matrix_chain[valid_pairs_ipSAE], axis=0)

        ipTM = np.max(ipTM_byres)
        ipSAE = np.max(ipSAE_byres)

        return ipTM, ipSAE

    def get_max_values(self) -> None:
        """
        Because some scores like ipSAE are not symmetric, meaning A->B != B->A, we
        take the maximal score for either direction to be the undirected score.
        Here we scrape through the internal dataframe and keeps only the rows with
        the maximal values.

        Returns:
            None
        """
        rows = []
        processed = set()
        for chain1, chain2 in self.permuted:
            if not all([chain in processed for chain in (chain1, chain2)]):
                filtered = self.df.filter(
                    ((pl.col('chain1') == chain1) & (pl.col('chain2') == chain2)) |
                    ((pl.col('chain1') == chain2) & (pl.col('chain2') == chain1))
                )
                max_ipsae = filtered.select('ipSAE').max().item()
                max_row = filtered.filter(pl.col('ipSAE') == max_ipsae)
                rows.append(max_row)

                processed.add(chain1)
                processed.add(chain2)

        self.scores = pl.concat(rows)

    def permute_chains(self) -> None:
        """
        Helper function that gives all permutations of chainID except
        the pair (self, self) for each chainID. This also ensures that
        if we have (A, B) we do not also store (B, A).

        Returns:
            None
        """
        permuted = set()
        for c1, c2 in permutations(self.unique_chains):
            if c1 != c2:
                permuted.add((c1, c2))
                permuted.add((c2, c1))

        self.permuted = list(permuted)

    @staticmethod
    def pDockQ_score(x) -> float:
        """
        Computes pDockQ score per the following equation.
        $pDockQ = \frac{0.724}{(1 + e^{-0.052 * (x - 152.611)}) + 0.018}$

        Details on the pDockQ score at: https://doi.org/10.1038/s41467-022-28865-w

        Arguments:
            x (float): Mean pLDDT score scaled by the log10 number of residue pairs 
                that meet pLDDT and distance cutoffs.

        Returns:
            (float): pDockQ score
        """
        return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

    @staticmethod
    def pDockQ2_score(x) -> float:
        """
        Computes pDockQ2 score per the following equation.
        $pDockQ = \frac{1.31}{(1 + e^{-0.075 * (x - 84.733)}) + 0.005}$

        Details on the pDockQ2 score at: https://doi.org/10.1093/bioinformatics/btad424

        Arguments:
            x (float): Mean pLDDT score scaled by mean PAE score.

        Returns:
            (float): pDockQ2 score
        """
        return 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005

    @staticmethod
    @vectorize
    def compute_pTM(x: float,
                    d0: float) -> float:
        """
        Computes pTM score per the following equation.
        $pTM = \frac{1.0}{(1 + (x / d0)^2)}$

        Arguments:
            x (float): pLDDT score
            d0 (float): d0 parameter

        Returns:
            (float): pTM score
        """
        return 1. / (1 + (x / d0) ** 2)

    @staticmethod
    def compute_d0(L: int,
                   pair_type: str) -> float:
        """
        Computes d0 term per the following equation. 
        $d0 = min(1.0, 1.24 * (L - 15)^}(\frac{1}{3})} - 1.8)$

        Arguments:
            L (int): Length of sequence up to 27 residues.
            pair_type (str): Whether or not chain is a nucleic acid.

        Returns:
            (float): d0
        """
        L = max(27, L)

        min_value = 1.
        if pair_type == 'nucleic_acid':
            min_value = 2.

        return max(min_value, 1.24 * (L - 15) ** (1/3) - 1.8)


class ModelParser:
    """
    Helper class to read in and process a structure file for downstream
    scoring tasks. Capable of reading both PDB and CIF formats.

    Arguments:
        structure (PathLike): Path to PDB or CIF file.
    """
    def __init__(self, 
                 structure: PathLike):
        self.structure =Path(structure)

        self.token_mask = []
        self.residues = []
        self.cb_residues = []
        self.chains = []

    def parse_structure_file(self) -> None:
        """
        Identify filetype, and parses line by line, storing relevant data
        for all C-alpha, C-beta and C1, C3 atoms for proteins and nucleic
        acids alike.

        Returns:
            None
        """
        if self.structure.suffix == '.pdb':
            line_parser = self.parse_pdb_line
        else:
            line_parser = self.parse_cif_line

        field_num = 0
        lines = open(self.structure).readlines()
        fields = dict()
        for line in lines:
            if line.startswith('_atom_site.'):
                _, field_name = line.strip().split('.')
                fields[field_name] = field_num
                field_num += 1

            if any([line.startswith(atom) for atom in ['ATOM', 'HETATM']]):
                atom = line_parser(line, fields)

                name = atom['atom_name']
                if name == 'CA':
                    self.token_mask.append(1)
                    self.residues.append(atom)
                    self.chains.append(atom['chain_id'])
                    if atom['res'] == 'GLY':
                        self.cb_residues.append(atom)

                elif 'C1' in name:
                    self.token_mask.append(1)
                    self.residues.append(atom)
                    self.chains.append(atom['chain_id'])

                elif name == 'CB' or 'C3' in name:
                    self.cb_residues.append(atom)

    def classify_chains(self) -> None:
        """
        Reads through residue data to assign the identity of each chain as
        either protein (by default) or nucleic acid if an NA residue is detected.

        Returns:
            None
        """
        self.residue_types = np.array([res['res'] for res in self.residues])
        chains = np.unique(self.chains)
        self.chain_types = {chain: 'protein' for chain in chains}
        for chain in chains:
            indices = np.where(chains == chain)[0]
            chain_residues = self.residue_types[indices]
            if any([r in chain_residues for r in self.nucleic_acids]):
                self.chain_types[chain] = 'nucleic_acid'

    @property
    def nucleic_acids(self) -> list[str]:
        """
        Stores the canonical resnames for RNA and DNA residues.

        Returns:
            (list[str]): List of nucleic acid resnames.
        """
        return ['DA', 'DC', 'DT', 'DG', 'A', 'C', 'U', 'G']

    @staticmethod
    def parse_pdb_line(line: str, 
                       *args) -> dict[str, Any]:
        """
        Parses a single line of a PDB file, extracting atom and residue information.
        Processes this into a dictionary and returns the dict.

        Arguments:
            line (str): Actual line from PDB file.
            *args: Just here so we can use the same API for PDB and CIF.

        Returns:
            (dict[str, Any]): Dictionary representation of data.
        """
        atom_num = line[6:11].strip()
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_id = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()

        return ModelParser.package_line(atom_num, atom_name, residue_name, chain_id, residue_id, x, y, z)

    @staticmethod
    def parse_cif_line(line: str, 
                       fields: dict[str, int]) -> dict[str, Any]:
        """
        Parses a single line of a CIF file, extracting atom and residue information.
        Processes this into a dictionary and returns the dict.

        Arguments:
            line (str): Actual line from CIF file.
            fields (dict[str, int]): Definition of where each field is found.

        Returns:
            (dict[str, Any]): Dictionary representation of data.
        """
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

        return ModelParser.package_line(atom_num, atom_name, residue_name, chain_id, residue_id, x, y, z)

    @staticmethod
    def package_line(atom_num: str,
                     atom_name: str,
                     residue_name: str,
                     chain_id: str,
                     residue_id: str,
                     x: str,
                     y: str,
                     z: str) -> dict[str, Any]:
        """
        Packs various information from a single line of a structure file into
        a dictionary to maintain consistency.

        Arguments:
            atom_num (str): Atom index.
            atom_name (str): Atom name.
            residue_name (str): Resname.
            chain_id (str): ChainID.
            residue_id (str): ResID.
            x (str): X coordinate.
            y (str): Y coordinate.
            z (str): Z coordinate.

        Returns:
            (dict[str, Any]): Dictionary representation of data.
        """
        return {
            'atom_num': int(atom_num),
            'atom_name': atom_name,
            'coor': np.array([float(i) for i in [x, y, z]]),
            'res': residue_name,
            'chain_id': chain_id,
            'resid': int(residue_id),
        }
