"""
Unit tests for sasa.py module
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import MDAnalysis as mda
from MDAnalysis.core import groups

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sasa import SASA, RelativeSASA


class TestSASA:
    """Test suite for SASA class"""
    
    @patch('sasa.KDTree')
    def test_init(self, mock_kdtree):
        """Test SASA initialization"""
        # Create mock universe and atomgroup
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O', 'N'])
        
        # Create SASA instance
        sasa = SASA(mock_ag, probe_radius=1.4, n_points=256)
        
        assert sasa.probe_radius == 1.4
        assert sasa.n_points == 256
        assert sasa.ag == mock_ag
        assert hasattr(sasa, 'radii')
        assert hasattr(sasa, 'sphere')
    
    def test_init_with_updating_atomgroup(self):
        """Test that UpdatingAtomGroup raises TypeError"""
        mock_ag = MagicMock(spec=groups.UpdatingAtomGroup)
        
        with pytest.raises(TypeError):
            SASA(mock_ag)
    
    def test_init_without_elements(self):
        """Test that missing elements raises ValueError"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        del mock_ag.elements  # Remove elements attribute
        
        with pytest.raises(ValueError):
            SASA(mock_ag)
    
    def test_get_sphere(self):
        """Test fibonacci sphere generation"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        
        sasa = SASA(mock_ag, n_points=100)
        sphere = sasa.get_sphere()
        
        assert sphere.shape == (100, 3)
        # Check that points are on unit sphere
        norms = np.linalg.norm(sphere, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)
    
    @patch('sasa.KDTree')
    def test_measure_sasa(self, mock_kdtree):
        """Test SASA measurement for atomgroup"""
        # Setup mock objects
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O'])
        mock_ag.n_atoms = 3
        mock_ag.positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        # Mock KDTree behavior
        mock_kdtree_instance = MagicMock()
        mock_kdtree_instance.query_ball_point.return_value = [0, 1]
        mock_kdtree.return_value = mock_kdtree_instance
        
        sasa = SASA(mock_ag, n_points=100)
        sasa.radii = np.array([1.7, 1.2, 1.52])
        sasa.points_available = set(range(100))
        
        result = sasa.measure_sasa(mock_ag)
        
        assert result.shape == (3,)
        assert all(r >= 0 for r in result)
    
    def test_prepare(self):
        """Test _prepare method"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        mock_ag.n_residues = 5
        
        sasa = SASA(mock_ag)
        sasa._prepare()
        
        assert hasattr(sasa.results, 'sasa')
        assert sasa.results.sasa.shape == (5,)
        assert all(s == 0 for s in sasa.results.sasa)
    
    @patch.object(SASA, 'measure_sasa')
    def test_single_frame(self, mock_measure):
        """Test _single_frame method"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        # Setup mock atoms with residue IDs
        mock_atoms = []
        for i in range(3):
            mock_atom = MagicMock()
            mock_atom.resid = i + 1
            mock_atoms.append(mock_atom)
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O'])
        mock_ag.n_residues = 3
        mock_ag.atoms = mock_atoms
        
        # Mock measure_sasa to return area values
        mock_measure.return_value = np.array([10.0, 20.0, 30.0])
        
        sasa = SASA(mock_ag)
        sasa.results = MagicMock()
        sasa.results.sasa = np.zeros(3)
        
        sasa._single_frame()
        
        mock_measure.assert_called_once_with(mock_ag)
        assert sasa.results.sasa[0] == 10.0
        assert sasa.results.sasa[1] == 20.0
        assert sasa.results.sasa[2] == 30.0
    
    def test_conclude(self):
        """Test _conclude method"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        
        sasa = SASA(mock_ag)
        sasa.results = MagicMock()
        sasa.results.sasa = np.array([100.0, 200.0, 300.0])
        sasa.n_frames = 10
        
        sasa._conclude()
        
        assert sasa.results.sasa[0] == 10.0
        assert sasa.results.sasa[1] == 20.0
        assert sasa.results.sasa[2] == 30.0


class TestRelativeSASA:
    """Test suite for RelativeSASA class"""
    
    def test_init_without_bonds(self):
        """Test that missing bonds raises ValueError"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        del mock_ag.bonds  # Remove bonds attribute
        
        with pytest.raises(ValueError):
            RelativeSASA(mock_ag)
    
    def test_prepare(self):
        """Test _prepare method for RelativeSASA"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        mock_ag.bonds = MagicMock()
        mock_ag.n_residues = 5
        
        rel_sasa = RelativeSASA(mock_ag)
        rel_sasa._prepare()
        
        assert hasattr(rel_sasa.results, 'sasa')
        assert hasattr(rel_sasa.results, 'relative_area')
        assert rel_sasa.results.sasa.shape == (5,)
        assert rel_sasa.results.relative_area.shape == (5,)
    
    @patch.object(RelativeSASA, 'measure_sasa')
    def test_single_frame_relative(self, mock_measure):
        """Test _single_frame method for RelativeSASA"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        # Setup mock residues
        mock_residues = MagicMock()
        mock_residues.resindices = np.array([0, 1, 2])
        mock_residues.resids = np.array([1, 2, 3])
        
        # Setup mock atoms
        mock_atoms = []
        for i in range(3):
            mock_atom = MagicMock()
            mock_atom.resid = i + 1
            mock_atoms.append(mock_atom)
        
        # Setup mock atomgroup
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H', 'O'])
        mock_ag.bonds = MagicMock()
        mock_ag.n_residues = 3
        mock_ag.atoms = mock_atoms
        mock_ag.residues = mock_residues
        
        # Mock select_atoms to return tripeptide
        mock_tripeptide = MagicMock()
        mock_tripeptide.__len__ = MagicMock(return_value=3)
        mock_tripeptide.resindices = np.array([0, 1, 2])
        mock_ag.select_atoms.return_value = mock_tripeptide
        
        # Mock measure_sasa to return different values for different calls
        mock_measure.side_effect = [
            np.array([10.0, 20.0, 30.0]),  # Initial SASA
            np.array([5.0, 10.0, 15.0]),   # Tripeptide SASA
            np.array([5.0, 10.0, 15.0]),   # Tripeptide SASA
            np.array([5.0, 10.0, 15.0]),   # Tripeptide SASA
        ]
        
        rel_sasa = RelativeSASA(mock_ag)
        rel_sasa.results = MagicMock()
        rel_sasa.results.sasa = np.zeros(3)
        rel_sasa.results.relative_area = np.zeros(3)
        
        rel_sasa._single_frame()
        
        assert mock_measure.called
    
    def test_conclude_relative(self):
        """Test _conclude method for RelativeSASA"""
        mock_universe = MagicMock()
        mock_trajectory = MagicMock()
        mock_universe.trajectory = mock_trajectory
        
        mock_ag = MagicMock(spec=mda.AtomGroup)
        mock_ag.universe = mock_universe
        mock_ag.elements = np.array(['C', 'H'])
        mock_ag.bonds = MagicMock()
        
        rel_sasa = RelativeSASA(mock_ag)
        rel_sasa.results = MagicMock()
        rel_sasa.results.sasa = np.array([100.0, 200.0, 300.0])
        rel_sasa.results.relative_area = np.array([0.5, 0.6, 0.7])
        rel_sasa.n_frames = 10
        
        rel_sasa._conclude()
        
        assert rel_sasa.results.sasa[0] == 10.0
        assert rel_sasa.results.sasa[1] == 20.0
        assert rel_sasa.results.sasa[2] == 30.0
        assert rel_sasa.results.relative_area[0] == 0.05
        assert rel_sasa.results.relative_area[1] == 0.06
        assert rel_sasa.results.relative_area[2] == 0.07


if __name__ == "__main__":
    pytest.main([__file__, "-v"])