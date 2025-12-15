"""Interface prediction Score from Aligned Errors (ipSAE) module.

This module computes interaction prediction scores from pLDDT and PAE data,
adapted from https://doi.org/10.1101/2025.02.10.637595. Supports outputs
from structure prediction tools like Boltz and AlphaFold.
"""

from itertools import permutations
import numpy as np
from numpy import vectorize
from pathlib import Path
import polars as pl
from typing import Any, Union

PathLike = Union[Path, str]
OptPath = Union[Path, str, None]


class ipSAE:
    """Compute interaction prediction Score from Aligned Errors.

    Computes various model quality scores including pDockQ, pDockQ2,
    LIS, ipTM, and ipSAE for structure predictions.

    Attributes:
        parser: ModelParser instance for structure file.
        plddt_file: Path to pLDDT data file.
        pae_file: Path to PAE data file.
        path: Output directory path.
        scores: Polars DataFrame of computed scores after run().

    Args:
        structure_file: Path to PDB/CIF model file.
        plddt_file: Path to pLDDT numpy file (.npz with 'plddt' key).
        pae_file: Path to PAE numpy file (.npz with 'pae' key).
        out_path: Output directory path. If None, uses parent directory
            of plddt_file.

    Example:
        >>> scorer = ipSAE('model.pdb', 'plddt.npz', 'pae.npz')
        >>> scorer.run()
        >>> print(scorer.scores)
    """

    def __init__(
        self, 
        structure_file: PathLike,
        plddt_file: PathLike,
        pae_file: PathLike,
        out_path: OptPath = None
    ):
        """Initialize the ipSAE scorer.

        Args:
            structure_file: Path to structure file.
            plddt_file: Path to pLDDT data file.
            pae_file: Path to PAE data file.
            out_path: Output directory path.
        """
        self.parser = ModelParser(structure_file)
        self.plddt_file = Path(plddt_file)
        self.pae_file = Path(pae_file)

        self.path = Path(out_path) if out_path is not None else self.plddt_file.parent
        self.path.mkdir(exist_ok=True)

    def parse_structure_file(self) -> None:
        """Parse the structure file and extract relevant details.

        Runs the parser to read the structure file and classifies
        chains as protein or nucleic acid.
        """
        self.parser.parse_structure_file()
        self.parser.classify_chains()
        self.coordinates = np.vstack([res['coor'] for res in self.parser.residues])
        self.token_array = np.array(self.parser.token_mask, dtype=bool)

    def prepare_scorer(self) -> None:
        """Initialize the ScoreCalculator for computing scores.

        Creates a ScoreCalculator instance with chain information
        extracted from the parsed structure.
        """
        chains = np.array(self.parser.chains)
        chain_types = self.parser.chain_types
        residue_types = np.array([res['res'] for res in self.parser.residues])

        self.scorer = ScoreCalculator(
            chains=chains,
            chain_pair_type=chain_types,
            n_residues=residue_types
        )

    def run(self) -> None:
        """Execute the complete ipSAE scoring workflow.

        Parses structure, computes distogram, loads pLDDT and PAE data,
        runs the scorer, and saves results.
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
        """Save scores DataFrame to a Parquet file."""
        self.scores.write_parquet(self.path / 'ipSAE_scores.parquet')

    def load_pLDDT_file(self) -> np.ndarray:
        """Load and scale pLDDT data.

        Returns:
            pLDDT array scaled to 0-100 range.
        """
        data = np.load(str(self.plddt_file))
        pLDDT_arr = np.array(data['plddt'] * 100.)
        return pLDDT_arr

    def load_PAE_file(self) -> np.ndarray:
        """Load PAE data from file.

        Returns:
            PAE array from the 'pae' key in the npz file.
        """
        data = np.load(str(self.pae_file))['pae']
        return data


class ScoreCalculator:
    """Calculate model quality scores from structure predictions.

    Computes pDockQ, pDockQ2, LIS, ipTM, and ipSAE scores for all
    chain pairs in a structure.

    Attributes:
        chains: Array of chain IDs for each residue.
        unique_chains: Unique chain IDs in the structure.
        chain_pair_type: Dictionary mapping chain ID to type.
        n_res: Array of residue types.
        permuted: List of all chain pairs to evaluate.
        scores: DataFrame of computed scores after compute_scores().

    Args:
        chains: Array of chain IDs.
        chain_pair_type: Dictionary mapping chain ID to chain type
            ('protein' or 'nucleic').
        n_residues: Number of residues per chain.
        pdockq_cutoff: Distance cutoff for pDockQ in Angstroms.
            Defaults to 8.0.
        pae_cutoff: PAE cutoff for ipSAE in Angstroms. Defaults to 12.0.
        dist_cutoff: General distance cutoff in Angstroms. Defaults to 10.0.

    Example:
        >>> calc = ScoreCalculator(chains, chain_types, n_residues)
        >>> calc.compute_scores(distances, plddt, pae)
        >>> print(calc.scores)
    """

    def __init__(
        self,
        chains: np.ndarray,
        chain_pair_type: dict[str, str],
        n_residues: int,
        pdockq_cutoff: float = 8.,
        pae_cutoff: float = 12.,
        dist_cutoff: float = 10.
    ):
        """Initialize the ScoreCalculator.

        Args:
            chains: Array of chain IDs.
            chain_pair_type: Chain ID to type mapping.
            n_residues: Residue type array.
            pdockq_cutoff: pDockQ distance cutoff.
            pae_cutoff: PAE cutoff.
            dist_cutoff: General distance cutoff.
        """
        self.chains = chains
        self.unique_chains = np.unique(chains)
        self.chain_pair_type = chain_pair_type
        self.n_res = n_residues
        self.pDockQ_cutoff = pdockq_cutoff
        self.PAE_cutoff = pae_cutoff
        self.dist_cutoff = dist_cutoff

        self.permute_chains()

    def compute_scores(
        self,
        distances: np.ndarray,
        pLDDT: np.ndarray,
        PAE: np.ndarray
    ) -> None:
        """Compute all scores for all chain pairs.

        Calculates pDockQ, pDockQ2, LIS, ipTM, and ipSAE scores for
        each permutation of chain pairs.

        Args:
            distances: Pairwise distance matrix between all residues.
            pLDDT: Per-residue pLDDT values (0-100 scale).
            PAE: Predicted aligned error matrix.
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

        self.df = pl.DataFrame(
            np.array(results), 
            schema={
                'chain1': str, 
                'chain2': str, 
                'pDockQ': float, 
                'pDockQ2': float,
                'LIS': float,
                'ipTM': float,
                'ipSAE': float
            }
        )
        self.get_max_values()

    def compute_pDockQ_scores(
        self,
        chain1: str,
        chain2: str
    ) -> tuple[float, float]:
        """Compute pDockQ and pDockQ2 scores for a chain pair.

        pDockQ depends solely on pLDDT, while pDockQ2 depends on both
        pLDDT and PAE.

        Args:
            chain1: First chain identifier.
            chain2: Second chain identifier.

        Returns:
            Tuple of (pDockQ, pDockQ2) scores.
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

    def compute_LIS(self, chain1: str, chain2: str) -> float:
        """Compute Local Interaction Score (LIS) for a chain pair.

        LIS is based on a subset of the predicted aligned error using
        a cutoff of 12 Å. Values range in (0, 1] where 1 indicates
        perfect accuracy.

        Adapted from: https://doi.org/10.1101/2024.02.19.580970

        Args:
            chain1: First chain identifier.
            chain2: Second chain identifier.

        Returns:
            LIS value for the chain pair.
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

    def compute_ipTM_ipSAE(
        self,
        chain1: str,
        chain2: str
    ) -> tuple[float, float]:
        """Compute ipTM and ipSAE scores for a chain pair.

        These operations are combined as they rely on similar
        data processing.

        Args:
            chain1: First chain identifier.
            chain2: Second chain identifier.

        Returns:
            Tuple of (ipTM, ipSAE) scores.
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
        """Extract maximum scores for undirected chain pairs.

        Because some scores like ipSAE are asymmetric (A->B != B->A),
        takes the maximum score for either direction as the undirected
        score.
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
        """Generate all permutations of chain pairs.

        Creates all unique ordered pairs of chains, excluding self-pairs.
        """
        permuted = set()
        for c1, c2 in permutations(self.unique_chains, 2):
            if c1 != c2:
                permuted.add((c1, c2))
                permuted.add((c2, c1))

        self.permuted = list(permuted)

    @staticmethod
    def pDockQ_score(x: float) -> float:
        """Compute pDockQ score.

        Formula: pDockQ = 0.724 / (1 + exp(-0.052 * (x - 152.611))) + 0.018

        Reference: https://doi.org/10.1038/s41467-022-28865-w

        Args:
            x: Mean pLDDT scaled by log10 of the number of residue pairs
                meeting pLDDT and distance cutoffs.

        Returns:
            pDockQ score.
        """
        return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

    @staticmethod
    def pDockQ2_score(x: float) -> float:
        """Compute pDockQ2 score.

        Formula: pDockQ2 = 1.31 / (1 + exp(-0.075 * (x - 84.733))) + 0.005

        Reference: https://doi.org/10.1093/bioinformatics/btad424

        Args:
            x: Mean pLDDT scaled by mean PAE score.

        Returns:
            pDockQ2 score.
        """
        return 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005

    @staticmethod
    @vectorize
    def compute_pTM(x: float, d0: float) -> float:
        """Compute pTM score.

        Formula: pTM = 1.0 / (1 + (x / d0)^2)

        Args:
            x: pLDDT or PAE value.
            d0: Distance parameter from compute_d0.

        Returns:
            pTM score.
        """
        return 1. / (1 + (x / d0) ** 2)

    @staticmethod
    def compute_d0(L: int, pair_type: str) -> float:
        """Compute d0 parameter for pTM calculation.

        Formula: d0 = max(min_value, 1.24 * (L - 15)^(1/3) - 1.8)

        Args:
            L: Sequence length (minimum 27).
            pair_type: 'protein' or 'nucleic_acid'.

        Returns:
            d0 parameter value.
        """
        L = max(27, L)

        min_value = 1.
        if pair_type == 'nucleic_acid':
            min_value = 2.

        return max(min_value, 1.24 * (L - 15) ** (1/3) - 1.8)


class ModelParser:
    """Parse structure files to extract residue and atom information.

    Handles both PDB and CIF format files, extracting C-alpha, C-beta,
    and nucleic acid backbone atom coordinates.

    Attributes:
        structure: Path to the structure file.
        token_mask: List of token indicators for each residue.
        residues: List of dictionaries containing residue information.
        cb_residues: List of C-beta residue dictionaries.
        chains: List of chain IDs for each residue.
        chain_types: Dictionary mapping chain ID to type after
            classify_chains().

    Args:
        structure: Path to PDB or CIF file.

    Example:
        >>> parser = ModelParser('model.pdb')
        >>> parser.parse_structure_file()
        >>> parser.classify_chains()
    """

    def __init__(self, structure: PathLike):
        """Initialize the ModelParser.

        Args:
            structure: Path to PDB or CIF file.
        """
        self.structure = Path(structure)

        self.token_mask = []
        self.residues = []
        self.cb_residues = []
        self.chains = []

    def parse_structure_file(self) -> None:
        """Parse the structure file and extract atom/residue data.

        Identifies file type and parses line by line, storing data for
        C-alpha, C-beta, C1', and C3' atoms.
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
        """Classify chains as protein or nucleic acid.

        Reads through residue data to assign chain identity based on
        whether nucleic acid residues are detected.
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
        """Get canonical nucleic acid residue names.

        Returns:
            List of RNA and DNA residue names.
        """
        return ['DA', 'DC', 'DT', 'DG', 'A', 'C', 'U', 'G']

    @staticmethod
    def parse_pdb_line(line: str, *args) -> dict[str, Any]:
        """Parse a single line of a PDB file.

        Args:
            line: Line from the PDB file.
            *args: Unused, for API compatibility with parse_cif_line.

        Returns:
            Dictionary with atom/residue information.
        """
        atom_num = line[6:11].strip()
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21]
        residue_id = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()

        return ModelParser.package_line(
            atom_num, atom_name, residue_name, 
            chain_id, residue_id, x, y, z
        )

    @staticmethod
    def parse_cif_line(line: str, fields: dict[str, int]) -> dict[str, Any]:
        """Parse a single line of a CIF file.

        Args:
            line: Line from the CIF file.
            fields: Dictionary mapping field names to column indices.

        Returns:
            Dictionary with atom/residue information, or None if
            residue_id is missing.
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

        return ModelParser.package_line(
            atom_num, atom_name, residue_name, 
            chain_id, residue_id, x, y, z
        )

    @staticmethod
    def package_line(
        atom_num: str,
        atom_name: str,
        residue_name: str,
        chain_id: str,
        residue_id: str,
        x: str,
        y: str,
        z: str
    ) -> dict[str, Any]:
        """Package parsed line data into a dictionary.

        Args:
            atom_num: Atom index.
            atom_name: Atom name (e.g., 'CA', 'CB').
            residue_name: Residue name (e.g., 'ALA').
            chain_id: Chain identifier.
            residue_id: Residue sequence number.
            x: X coordinate as string.
            y: Y coordinate as string.
            z: Z coordinate as string.

        Returns:
            Dictionary containing parsed atom/residue data.
        """
        return {
            'atom_num': int(atom_num),
            'atom_name': atom_name,
            'coor': np.array([float(i) for i in [x, y, z]]),
            'res': residue_name,
            'chain_id': chain_id,
            'resid': int(residue_id),
        }
