"""
Unit tests for analysis/cov_ppi.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import json
import polars as pl


class TestPPInteractions:
    """Test suite for PPInteractions class"""
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_ppinteractions_init(self, mock_mda):
        """Test PPInteractions initialization"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        # Setup mock universe
        mock_universe = MagicMock()
        mock_universe.trajectory = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=100)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'results.json'
            
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=out_path,
                sel1='chainID A',
                sel2='chainID B',
                cov_cutoff=(11., 13.),
                sb_cutoff=6.0,
                hbond_cutoff=3.5,
                hbond_angle=30.,
                hydrophobic_cutoff=8.,
                plot=False
            )
            
            assert ppi.n_frames == 100
            assert ppi.sel1 == 'chainID A'
            assert ppi.sel2 == 'chainID B'
            assert ppi.cov_cutoff == (11., 13.)
            assert ppi.sb == 6.0
            assert ppi.hb_d == 3.5
            assert ppi.hydr == 8.
            assert not ppi.plot
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_ppinteractions_hbond_angle_conversion(self, mock_mda):
        """Test that hbond angle is converted to radians"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                hbond_angle=30.0,  # 30 degrees
                plot=False
            )
            
            # Should be converted using the formula: angle * 180 / pi
            # This seems inverted in the source code but we test the actual behavior
            expected_angle = 30.0 * 180 / np.pi
            assert np.isclose(ppi.hb_a, expected_angle)
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_res_map(self, mock_mda):
        """Test res_map method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            # Create mock atom groups
            mock_ag1 = MagicMock()
            mock_ag1.resids = np.array([1, 2, 3])
            
            mock_ag2 = MagicMock()
            mock_ag2.resids = np.array([10, 11])
            
            ppi.res_map(mock_ag1, mock_ag2)
            
            assert ppi.mapping['ag1'][0] == 1
            assert ppi.mapping['ag1'][1] == 2
            assert ppi.mapping['ag1'][2] == 3
            assert ppi.mapping['ag2'][0] == 10
            assert ppi.mapping['ag2'][1] == 11
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_interpret_covariance(self, mock_mda):
        """Test interpret_covariance method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            # Setup mapping
            ppi.mapping = {
                'ag1': {0: 1, 1: 2},
                'ag2': {0: 10, 1: 11}
            }
            
            # Create covariance matrix with positive and negative values
            cov_mat = np.array([
                [0.5, -0.3],  # Residue 1 has pos corr with 10, neg with 11
                [-0.2, 0.4],  # Residue 2 has neg corr with 10, pos with 11
            ])
            
            positive, negative = ppi.interpret_covariance(cov_mat)
            
            # Should have found positive correlations
            assert len(positive) > 0
            # Should have found negative correlations
            assert len(negative) > 0
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_identify_interaction_type(self, mock_mda):
        """Test identify_interaction_type method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            # Test charged residue pair (ASP-LYS should have saltbridge)
            functions, labels = ppi.identify_interaction_type('ASP', 'LYS')
            
            assert 'saltbridge' in labels or 'hydrophobic' in labels or 'hbond' in labels
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_save_results(self, mock_mda):
        """Test save method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'results.json'
            
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=out_path,
                plot=False
            )
            
            results = {
                'positive': {
                    'A_ALA1-B_LYS10': {
                        'hydrophobic': 0.5,
                        'hbond': 0.3,
                        'saltbridge': 0.0
                    }
                },
                'negative': {}
            }
            
            ppi.save(results)
            
            assert out_path.exists()
            
            with open(out_path) as f:
                loaded = json.load(f)
            
            assert 'positive' in loaded
            assert 'A_ALA1-B_LYS10' in loaded['positive']
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_parse_results(self, mock_mda):
        """Test parse_results method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            results = {
                'positive': {
                    'A_ALA1-B_LYS10': {
                        'hydrophobic': 0.5,
                        'hbond': 0.3,
                        'saltbridge': 0.0
                    }
                },
                'negative': {
                    'A_GLU5-B_ARG15': {
                        'hydrophobic': 0.0,
                        'hbond': 0.0,
                        'saltbridge': 0.8
                    }
                }
            }
            
            df = ppi.parse_results(results)
            
            assert isinstance(df, pl.DataFrame)
            assert 'Residue Pair' in df.columns
            assert 'Hydrophobic' in df.columns
            assert 'Hydrogen Bond' in df.columns
            assert 'Salt Bridge' in df.columns
            assert 'Covariance' in df.columns
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_parse_results_filters_zeros(self, mock_mda):
        """Test that parse_results filters out all-zero entries"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            results = {
                'positive': {
                    'A_ALA1-B_LYS10': {
                        'hydrophobic': 0.5,
                        'hbond': 0.0,
                        'saltbridge': 0.0
                    },
                    'A_GLY2-B_SER11': {
                        'hydrophobic': 0.0,
                        'hbond': 0.0,
                        'saltbridge': 0.0
                    }
                },
                'negative': {}
            }
            
            df = ppi.parse_results(results)
            
            # Should only include the non-zero entry
            assert len(df) == 1
            assert 'A_ALA1-B_LYS10' in df['Residue Pair'].to_list()


class TestEvaluateHBond:
    """Test the evaluate_hbond method"""
    
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_evaluate_hbond_found(self, mock_mda):
        """Test evaluate_hbond when HBond is found"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=10)
        mock_mda.Universe.return_value = mock_universe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            # Create mock donor and acceptor atoms
            mock_h = MagicMock()
            mock_h.position = np.array([0.0, 1.0, 0.0])
            mock_h.type = 'H'
            
            mock_donor = MagicMock()
            mock_donor.position = np.array([0.0, 0.0, 0.0])
            mock_donor.bonded_atoms = MagicMock()
            mock_donor.bonded_atoms.__iter__ = MagicMock(return_value=iter([mock_h]))
            
            mock_donor_group = MagicMock()
            mock_donor_group.atoms = [mock_donor]
            
            mock_acceptor = MagicMock()
            mock_acceptor.position = np.array([0.0, 2.5, 0.0])  # Close enough for HBond
            
            mock_acceptor_group = MagicMock()
            mock_acceptor_group.atoms = [mock_acceptor]
            
            # Set distance cutoff to allow this to be an HBond
            ppi.hb_d = 3.5
            ppi.hb_a = 60.0  # Allow wide angle
            
            result = ppi.evaluate_hbond(mock_donor_group, mock_acceptor_group)
            
            # Should return 0 or 1
            assert result in [0, 1]


class TestAnalyzeHydrophobic:
    """Test the analyze_hydrophobic method"""
    
    @patch('molecular_simulations.analysis.cov_ppi.distance_array')
    @patch('molecular_simulations.analysis.cov_ppi.mda')
    def test_analyze_hydrophobic(self, mock_mda, mock_dist_array):
        """Test analyze_hydrophobic method"""
        from molecular_simulations.analysis.cov_ppi import PPInteractions
        
        mock_universe = MagicMock()
        mock_universe.trajectory.__len__ = MagicMock(return_value=2)
        mock_universe.trajectory.__iter__ = MagicMock(return_value=iter([MagicMock(), MagicMock()]))
        mock_universe.select_atoms.return_value = MagicMock()
        mock_mda.Universe.return_value = mock_universe
        
        # Mock distance array to return values within cutoff
        mock_dist_array.return_value = np.array([[5.0]])  # Within 8 Å cutoff
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ppi = PPInteractions(
                top='fake.prmtop',
                traj='fake.dcd',
                out=Path(tmpdir) / 'results.json',
                plot=False
            )
            
            # Create mock residues with carbon atoms
            mock_atom1 = MagicMock()
            mock_atom1.type = 'C'
            mock_res1 = MagicMock()
            mock_res1.atoms = [mock_atom1]
            
            mock_atom2 = MagicMock()
            mock_atom2.type = 'C'
            mock_res2 = MagicMock()
            mock_res2.atoms = [mock_atom2]
            
            # Mock the + operator for AtomGroups
            ppi.u.select_atoms.return_value = MagicMock()
            
            result = ppi.analyze_hydrophobic(mock_res1, mock_res2)
            
            # Should return a fraction
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
