"""
Unit tests for analysis/utils.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import shutil


class TestEmbedData:
    """Test suite for EmbedData class"""
    
    @patch('molecular_simulations.analysis.utils.mda')
    def test_embed_data_init(self, mock_mda):
        """Test EmbedData initialization"""
        from molecular_simulations.analysis.utils import EmbedData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            mock_universe = MagicMock()
            mock_mda.Universe.return_value = mock_universe
            
            embedding_dict = {'all': np.array([1.0, 2.0, 3.0])}
            embedder = EmbedData(pdb_path, embedding_dict)
            
            assert embedder.pdb == pdb_path
            assert embedder.out == pdb_path  # No output specified
            assert embedder.embeddings == embedding_dict
    
    @patch('molecular_simulations.analysis.utils.mda')
    def test_embed_data_with_output(self, mock_mda):
        """Test EmbedData with custom output path"""
        from molecular_simulations.analysis.utils import EmbedData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            out_path = Path(tmpdir) / 'output.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            mock_universe = MagicMock()
            mock_mda.Universe.return_value = mock_universe
            
            embedding_dict = {'all': np.array([1.0, 2.0, 3.0])}
            embedder = EmbedData(pdb_path, embedding_dict, out=out_path)
            
            assert embedder.out == out_path
    
    @patch('molecular_simulations.analysis.utils.mda')
    def test_embed_selection(self, mock_mda):
        """Test embed_selection method"""
        from molecular_simulations.analysis.utils import EmbedData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            # Create mock universe with residues
            mock_residue1 = MagicMock()
            mock_residue1.atoms = MagicMock()
            mock_residue1.atoms.tempfactors = np.zeros(5)
            
            mock_residue2 = MagicMock()
            mock_residue2.atoms = MagicMock()
            mock_residue2.atoms.tempfactors = np.zeros(5)
            
            mock_selection = MagicMock()
            mock_selection.residues = [mock_residue1, mock_residue2]
            
            mock_universe = MagicMock()
            mock_universe.select_atoms.return_value = mock_selection
            mock_mda.Universe.return_value = mock_universe
            
            embedding_dict = {'protein': np.array([1.5, 2.5])}
            embedder = EmbedData(pdb_path, embedding_dict)
            
            embedder.embed_selection('protein', np.array([1.5, 2.5]))
            
            # Check selection was called
            mock_universe.select_atoms.assert_called_with('protein')
    
    @patch('molecular_simulations.analysis.utils.mda')
    @patch('molecular_simulations.analysis.utils.shutil')
    def test_write_new_pdb_with_backup(self, mock_shutil, mock_mda):
        """Test write_new_pdb creates backup when output exists"""
        from molecular_simulations.analysis.utils import EmbedData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            mock_universe = MagicMock()
            mock_mda.Universe.return_value = mock_universe
            mock_writer = MagicMock()
            mock_mda.Writer.return_value.__enter__ = MagicMock(return_value=mock_writer)
            mock_mda.Writer.return_value.__exit__ = MagicMock(return_value=False)
            
            embedding_dict = {'all': np.array([1.0])}
            embedder = EmbedData(pdb_path, embedding_dict)
            
            embedder.write_new_pdb()
            
            # Should have called Writer
            mock_mda.Writer.assert_called_once()
    
    @patch('molecular_simulations.analysis.utils.mda')
    def test_embed_method(self, mock_mda):
        """Test the full embed workflow"""
        from molecular_simulations.analysis.utils import EmbedData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            # Create mock for universe
            mock_residue = MagicMock()
            mock_residue.atoms = MagicMock()
            mock_residue.atoms.tempfactors = np.zeros(5)
            
            mock_selection = MagicMock()
            mock_selection.residues = [mock_residue]
            
            mock_universe = MagicMock()
            mock_universe.select_atoms.return_value = mock_selection
            mock_mda.Universe.return_value = mock_universe
            mock_mda.Writer.return_value.__enter__ = MagicMock()
            mock_mda.Writer.return_value.__exit__ = MagicMock(return_value=False)
            
            embedding_dict = {'protein': np.array([1.5])}
            embedder = EmbedData(pdb_path, embedding_dict)
            
            # Should run without error
            embedder.embed()


class TestEmbedEnergyData:
    """Test suite for EmbedEnergyData class"""
    
    @patch('molecular_simulations.analysis.utils.mda')
    def test_embed_energy_data_init(self, mock_mda):
        """Test EmbedEnergyData initialization and preprocessing"""
        from molecular_simulations.analysis.utils import EmbedEnergyData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            mock_universe = MagicMock()
            mock_mda.Universe.return_value = mock_universe
            
            # Energy data with LJ and coulombic terms
            energy_data = np.array([
                [[1.0, -0.5], [2.0, -1.0]],  # Frame 1: 2 residues, 2 terms
                [[1.5, -0.7], [2.5, -1.2]],  # Frame 2
            ])
            
            embedding_dict = {'protein': energy_data}
            embedder = EmbedEnergyData(pdb_path, embedding_dict)
            
            # Preprocessed data should be different from input
            assert 'protein' in embedder.embeddings
    
    def test_sanitize_data_3d(self):
        """Test sanitize_data with 3D input (n_frames, n_residues, n_terms)"""
        from molecular_simulations.analysis.utils import EmbedEnergyData
        
        # 3D data: (2 frames, 3 residues, 2 terms)
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]],
        ])
        
        result = EmbedEnergyData.sanitize_data(data)
        
        # Should average over frames, then sum terms
        # Average: [[1.25, 2.25], [3.25, 4.25], [5.25, 6.25]]
        # Sum: [3.5, 7.5, 11.5]
        expected = np.array([3.5, 7.5, 11.5])
        
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sanitize_data_2d(self):
        """Test sanitize_data with 2D input (n_residues, n_terms)"""
        from molecular_simulations.analysis.utils import EmbedEnergyData
        
        # 2D data: (3 residues, 2 terms)
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        
        result = EmbedEnergyData.sanitize_data(data)
        
        # Should sum terms
        expected = np.array([3.0, 7.0, 11.0])
        
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, expected)
    
    @patch('molecular_simulations.analysis.utils.mda')
    def test_preprocess_rescaling(self, mock_mda):
        """Test that preprocessing rescales data appropriately"""
        from molecular_simulations.analysis.utils import EmbedEnergyData
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'test.pdb'
            pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            mock_universe = MagicMock()
            mock_mda.Universe.return_value = mock_universe
            
            # Simple energy data
            energy_data = np.array([
                [[-10.0, -5.0]],  # negative total = -15
                [[-20.0, -10.0]],  # negative total = -30
            ])
            
            embedding_dict = {'sel1': energy_data}
            embedder = EmbedEnergyData(pdb_path, embedding_dict)
            
            # All rescaled values should be <= 1
            for sel, data in embedder.embeddings.items():
                assert np.all(data <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
