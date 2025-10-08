"""
Unit tests for ipSAE.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import polars as pl

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ipSAE import ipSAE, ScoreCalculator, ModelParser


class TestModelParser:
    """Test suite for ModelParser class"""
    
    def test_init(self):
        """Test ModelParser initialization"""
        parser = ModelParser("test.pdb")
        assert parser.structure == Path("test.pdb")
        assert parser.token_mask == []
        assert parser.residues == []
        assert parser.cb_residues == []
        assert parser.chains == []
    
    def test_nucleic_acids_property(self):
        """Test nucleic acids property"""
        parser = ModelParser("test.pdb")
        na_list = parser.nucleic_acids
        assert 'DA' in na_list
        assert 'DC' in na_list
        assert 'DT' in na_list
        assert 'DG' in na_list
        assert 'A' in na_list
        assert 'C' in na_list
        assert 'U' in na_list
        assert 'G' in na_list
    
    def test_parse_pdb_line(self):
        """Test PDB line parsing"""
        pdb_line = "ATOM      1  CA  GLY A   1      10.000  20.000  30.000  1.00 20.00           C"
        result = ModelParser.parse_pdb_line(pdb_line)
        
        assert result['atom_num'] == 1
        assert result['atom_name'] == 'CA'
        assert result['res'] == 'GLY'
        assert result['chain_id'] == 'A'
        assert result['resid'] == 1
        assert np.allclose(result['coor'], [10.0, 20.0, 30.0])
    
    def test_parse_cif_line(self):
        """Test CIF line parsing"""
        cif_line = "ATOM 1 CA CA GLY A 1 1 10.000 20.000 30.000 1.00 20.00 C"
        fields = {
            'id': 0,
            'label_atom_id': 2,
            'label_comp_id': 3,
            'label_asym_id': 4,
            'label_seq_id': 6,
            'Cartn_x': 7,
            'Cartn_y': 8,
            'Cartn_z': 9
        }
        
        result = ModelParser.parse_cif_line(cif_line, fields)
        
        assert result['atom_num'] == 1
        assert result['atom_name'] == 'CA'
        assert result['res'] == 'GLY'
        assert result['chain_id'] == 'A'
        assert result['resid'] == 1
        assert np.allclose(result['coor'], [10.0, 20.0, 30.0])
    
    def test_parse_cif_line_missing_resid(self):
        """Test CIF line parsing with missing residue ID"""
        cif_line = "ATOM 1 CA CA GLY A . . 10.000 20.000 30.000 1.00 20.00 C"
        fields = {
            'id': 0,
            'label_atom_id': 2,
            'label_comp_id': 3,
            'label_asym_id': 4,
            'label_seq_id': 5,
            'Cartn_x': 7,
            'Cartn_y': 8,
            'Cartn_z': 9
        }
        
        result = ModelParser.parse_cif_line(cif_line, fields)
        assert result is None
    
    @patch("builtins.open", mock_open(read_data="""ATOM      1  CA  GLY A   1      10.000  20.000  30.000  1.00 20.00           C
ATOM      2  CB  GLY A   1      11.000  21.000  31.000  1.00 20.00           C
ATOM      3  CA  ALA A   2      12.000  22.000  32.000  1.00 20.00           C
END"""))
    def test_parse_structure_file_pdb(self):
        """Test parsing PDB structure file"""
        parser = ModelParser("test.pdb")
        parser.parse_structure_file()
        
        assert len(parser.residues) == 2  # Only CA atoms
        assert len(parser.token_mask) == 2
        assert len(parser.chains) == 2
        assert all(chain == 'A' for chain in parser.chains)
    
    def test_classify_chains(self):
        """Test chain classification"""
        parser = ModelParser("test.pdb")
        parser.residue_types = np.array(['GLY', 'ALA', 'DA', 'DC'])
        parser.chains = ['A', 'A', 'B', 'B']
        
        parser.classify_chains()
        
        assert parser.chain_types['A'] == 'protein'
        assert parser.chain_types['B'] == 'nucleic_acid'


class TestScoreCalculator:
    """Test suite for ScoreCalculator class"""
    
    def test_init(self):
        """Test ScoreCalculator initialization"""
        chains = np.array(['A', 'A', 'B', 'B'])
        chain_pair_type = {'A': 'protein', 'B': 'protein'}
        n_residues = 4
        
        calc = ScoreCalculator(chains, chain_pair_type, n_residues)
        
        assert np.array_equal(calc.chains, chains)
        assert np.array_equal(calc.unique_chains, np.array(['A', 'B']))
        assert calc.n_res == n_residues
        assert calc.pDockQ_cutoff == 8.0
        assert calc.PAE_cutoff == 12.0
        assert calc.dist_cutoff == 10.0
    
    def test_pDockQ_score(self):
        """Test pDockQ score calculation"""
        score = ScoreCalculator.pDockQ_score(150.0)
        assert 0 <= score <= 1
        
        # Test with extreme values
        score_low = ScoreCalculator.pDockQ_score(0.0)
        score_high = ScoreCalculator.pDockQ_score(300.0)
        assert score_low < score_high
    
    def test_pDockQ2_score(self):
        """Test pDockQ2 score calculation"""
        score = ScoreCalculator.pDockQ2_score(80.0)
        assert 0 <= score <= 1.5
        
        # Test with extreme values
        score_low = ScoreCalculator.pDockQ2_score(0.0)
        score_high = ScoreCalculator.pDockQ2_score(200.0)
        assert score_low < score_high
    
    def test_compute_pTM(self):
        """Test pTM score calculation"""
        x = np.array([5.0, 10.0, 15.0])
        d0 = 10.0
        scores = ScoreCalculator.compute_pTM(x, d0)
        
        assert scores.shape == (3,)
        assert all(0 <= s <= 1 for s in scores)
    
    def test_compute_d0(self):
        """Test d0 calculation"""
        # Test with small L (should return min_value)
        d0_small = ScoreCalculator.compute_d0(10, 'protein')
        assert d0_small == 1.0
        
        # Test with larger L
        d0_large = ScoreCalculator.compute_d0(100, 'protein')
        assert d0_large > 1.0
        
        # Test nucleic acid
        d0_na = ScoreCalculator.compute_d0(10, 'nucleic_acid')
        assert d0_na == 2.0
    
    def test_permute_chains(self):
        """Test chain permutation"""
        chains = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        chain_pair_type = {'A': 'protein', 'B': 'protein', 'C': 'protein'}
        
        calc = ScoreCalculator(chains, chain_pair_type, 6)
        
        # Should have permutations for all pairs except self-pairs
        assert ('A', 'B') in calc.permuted or ('B', 'A') in calc.permuted
        assert ('A', 'C') in calc.permuted or ('C', 'A') in calc.permuted
        assert ('B', 'C') in calc.permuted or ('C', 'B') in calc.permuted
        assert ('A', 'A') not in calc.permuted
    
    @patch('polars.DataFrame.write_parquet')
    def test_compute_scores(self, mock_write):
        """Test score computation"""
        chains = np.array(['A', 'A', 'B', 'B'])
        chain_pair_type = {'A': 'protein', 'B': 'protein'}
        
        calc = ScoreCalculator(chains, chain_pair_type, 4)
        
        # Create mock data
        distances = np.random.rand(4, 4) * 20
        pLDDT = np.random.rand(4) * 100
        PAE = np.random.rand(4, 4) * 15
        
        calc.compute_scores(distances, pLDDT, PAE)
        
        assert hasattr(calc, 'df')
        assert hasattr(calc, 'scores')


class TestIpSAE:
    """Test suite for ipSAE class"""
    
    @patch('ipSAE.ModelParser')
    def test_init(self, mock_parser):
        """Test ipSAE initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy files
            structure_file = Path(tmpdir) / "test.pdb"
            structure_file.touch()
            plddt_file = Path(tmpdir) / "plddt.npy"
            pae_file = Path(tmpdir) / "pae.npy"
            
            ipsae = ipSAE(structure_file, plddt_file, pae_file)
            
            assert ipsae.plddt_file == plddt_file
            assert ipsae.pae_file == pae_file
            assert ipsae.path == tmpdir
    
    def test_load_pLDDT_file(self):
        """Test loading pLDDT file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test pLDDT data
            plddt_data = np.random.rand(100) * 0.01  # Values between 0 and 1
            plddt_file = Path(tmpdir) / "plddt.npz"
            np.savez(plddt_file, plddt=plddt_data)
            
            ipsae = ipSAE("test.pdb", plddt_file, "pae.npy")
            result = ipsae.load_pLDDT_file()
            
            # Check that values are scaled by 100
            assert np.allclose(result, plddt_data * 100)
    
    def test_load_PAE_file(self):
        """Test loading PAE file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test PAE data
            pae_data = np.random.rand(100, 100) * 30
            pae_file = Path(tmpdir) / "pae.npz"
            np.savez(pae_file, pae=pae_data)
            
            ipsae = ipSAE("test.pdb", "plddt.npy", pae_file)
            result = ipsae.load_PAE_file()
            
            assert np.array_equal(result, pae_data)
    
    @patch('polars.DataFrame.write_parquet')
    def test_save_scores(self, mock_write):
        """Test saving scores to parquet"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipsae = ipSAE("test.pdb", "plddt.npy", "pae.npy", out_path=tmpdir)
            ipsae.scores = pl.DataFrame({'test': [1, 2, 3]})
            
            ipsae.save_scores()
            
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            assert 'ipSAE_scores.parquet' in str(args[0])
    
    @patch.object(ModelParser, 'parse_structure_file')
    @patch.object(ModelParser, 'classify_chains')
    def test_parse_structure_file(self, mock_classify, mock_parse):
        """Test structure file parsing"""
        # Setup mock parser
        mock_parser = MagicMock()
        mock_parser.residues = [
            {'coor': np.array([0, 0, 0])},
            {'coor': np.array([1, 1, 1])},
            {'coor': np.array([2, 2, 2])}
        ]
        mock_parser.token_mask = [1, 1, 1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ipsae = ipSAE("test.pdb", "plddt.npy", "pae.npy", out_path=tmpdir)
            ipsae.parser = mock_parser
            
            ipsae.parse_structure_file()
            
            assert ipsae.coordinates.shape == (3, 3)
            assert ipsae.token_array.shape == (3,)
            mock_parse.assert_called_once()
            mock_classify.assert_called_once()
    
    def test_prepare_scorer(self):
        """Test scorer preparation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ipsae = ipSAE("test.pdb", "plddt.npy", "pae.npy", out_path=tmpdir)
            
            # Setup mock parser data
            ipsae.parser.chains = ['A', 'A', 'B']
            ipsae.parser.chain_types = {'A': 'protein', 'B': 'protein'}
            ipsae.parser.residues = [
                {'res': 'GLY'},
                {'res': 'ALA'},
                {'res': 'VAL'}
            ]
            
            ipsae.prepare_scorer()
            
            assert isinstance(ipsae.scorer, ScoreCalculator)
            assert np.array_equal(ipsae.scorer.chains, ['A', 'A', 'B'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
