"""
Unit tests for autocluster.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import polars as pl

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autocluster import GenericDataloader, PeriodicDataloader, AutoKMeans, Decomposition


class TestGenericDataloader:
    """Test suite for GenericDataloader class"""
    
    def test_init(self):
        """Test initialization of GenericDataloader"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test numpy files
            test_data1 = np.random.rand(10, 5)
            test_data2 = np.random.rand(15, 5)
            
            file1 = Path(tmpdir) / "data1.npy"
            file2 = Path(tmpdir) / "data2.npy"
            
            np.save(file1, test_data1)
            np.save(file2, test_data2)
            
            # Initialize dataloader
            loader = GenericDataloader([file1, file2])
            
            # Check that data is loaded correctly
            assert loader.data.shape == (25, 5)
            assert len(loader.shapes) == 2
            assert loader.shapes[0] == (10, 5)
            assert loader.shapes[1] == (15, 5)
    
    def test_data_property(self):
        """Test data property returns correct array"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / "data.npy"
            np.save(file, test_data)
            
            loader = GenericDataloader([file])
            assert np.array_equal(loader.data, test_data)
    
    def test_shape_property_uniform(self):
        """Test shape property when all files have same shape"""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                data = np.random.rand(10, 5)
                file = Path(tmpdir) / f"data{i}.npy"
                np.save(file, data)
                files.append(file)
            
            loader = GenericDataloader(files)
            assert loader.shape == (10, 5)
    
    def test_shape_property_mixed(self):
        """Test shape property when files have different shapes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = np.random.rand(10, 5)
            data2 = np.random.rand(15, 5)
            
            file1 = Path(tmpdir) / "data1.npy"
            file2 = Path(tmpdir) / "data2.npy"
            
            np.save(file1, data1)
            np.save(file2, data2)
            
            loader = GenericDataloader([file1, file2])
            assert loader.shape == [(10, 5), (15, 5)]
    
    def test_multidimensional_reshape(self):
        """Test reshaping of multidimensional data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 3, 4)
            file = Path(tmpdir) / "data.npy"
            np.save(file, test_data)
            
            loader = GenericDataloader([file])
            assert loader.data.shape == (10, 12)  # 3*4 = 12


class TestPeriodicDataloader:
    """Test suite for PeriodicDataloader class"""
    
    def test_remove_periodicity(self):
        """Test periodicity removal using sin/cos decomposition"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with known values
            test_data = np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]])
            file = Path(tmpdir) / "periodic.npy"
            np.save(file, test_data)
            
            loader = PeriodicDataloader([file])
            
            # Check that features are doubled
            assert loader.data.shape[1] == test_data.shape[1] * 2
            
            # Verify sin/cos decomposition
            # For angle 0: cos(0)=1, sin(0)=0
            # For angle π/2: cos(π/2)=0, sin(π/2)=1
            assert np.isclose(loader.data[0, 0], 1.0)  # cos(0)
            assert np.isclose(loader.data[0, 1], 0.0)  # sin(0)
            assert np.isclose(loader.data[0, 2], 0.0, atol=1e-10)  # cos(π/2)
            assert np.isclose(loader.data[0, 3], 1.0)  # sin(π/2)


class TestAutoKMeans:
    """Test suite for AutoKMeans class"""
    
    @patch('autocluster.KMeans')
    @patch('autocluster.silhouette_score')
    def test_sweep_n_clusters(self, mock_silhouette, mock_kmeans):
        """Test n_clusters parameter sweep"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            test_data = np.random.rand(20, 5)
            file = Path(tmpdir) / "data.npy"
            np.save(file, test_data)
            
            # Setup mocks
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 1] * 10)
            mock_kmeans_instance.cluster_centers_ = np.array([[0, 0], [1, 1]])
            mock_kmeans.return_value = mock_kmeans_instance
            
            mock_silhouette.return_value = 0.5
            
            # Initialize AutoKMeans
            auto_km = AutoKMeans(tmpdir, max_clusters=5, stride=1)
            auto_km.reduced = np.random.rand(20, 2)
            
            # Run sweep
            auto_km.sweep_n_clusters([2, 3, 4])
            
            # Check that silhouette score was called
            assert mock_silhouette.called
            assert mock_kmeans.called
    
    def test_map_centers_to_frames(self):
        """Test mapping cluster centers to frames"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / "data.npy"
            np.save(file, test_data)
            
            auto_km = AutoKMeans(tmpdir)
            auto_km.reduced = np.array([[0, 0], [1, 1], [2, 2]])
            auto_km.centers = np.array([[0.1, 0.1], [1.1, 1.1]])
            auto_km.shape = (3, 5)
            
            auto_km.map_centers_to_frames()
            
            assert len(auto_km.cluster_centers) == 2
            assert all(isinstance(v, tuple) for v in auto_km.cluster_centers.values())
    
    @patch('autocluster.json.dump')
    def test_save_centers(self, mock_json_dump):
        """Test saving cluster centers to JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / "data.npy"
            np.save(file, test_data)
            
            auto_km = AutoKMeans(tmpdir)
            auto_km.cluster_centers = {0: (0, 1), 1: (0, 2)}
            
            auto_km.save_centers()
            
            # Check that json.dump was called
            assert mock_json_dump.called
    
    @patch('polars.DataFrame.write_parquet')
    def test_save_labels(self, mock_write_parquet):
        """Test saving cluster labels to parquet"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(10, 5)
            file = Path(tmpdir) / "data.npy"
            np.save(file, test_data)
            
            auto_km = AutoKMeans(tmpdir)
            auto_km.dataloader.files = [Path("test1.npy"), Path("test2.npy")]
            auto_km.dataloader.shape = (5, 3)
            auto_km.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            
            auto_km.save_labels()
            
            # Check that write_parquet was called
            assert mock_write_parquet.called


class TestDecomposition:
    """Test suite for Decomposition class"""
    
    def test_pca_initialization(self):
        """Test PCA decomposition initialization"""
        decomp = Decomposition('PCA', n_components=2)
        assert decomp.decomposer is not None
    
    def test_fit_transform(self):
        """Test fit_transform method"""
        X = np.random.rand(100, 10)
        decomp = Decomposition('PCA', n_components=2)
        
        X_reduced = decomp.fit_transform(X)
        
        assert X_reduced.shape == (100, 2)
    
    def test_separate_fit_and_transform(self):
        """Test separate fit and transform methods"""
        X = np.random.rand(100, 10)
        decomp = Decomposition('PCA', n_components=2)
        
        decomp.fit(X)
        X_reduced = decomp.transform(X)
        
        assert X_reduced.shape == (100, 2)
    
    def test_unsupported_algorithm(self):
        """Test that unsupported algorithms raise appropriate errors"""
        with pytest.raises(TypeError):
            # TICA and UMAP are not implemented (None in the dict)
            decomp = Decomposition('TICA')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
