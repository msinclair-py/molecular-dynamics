"""
Unit tests for fingerprinter.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import MDAnalysis as mda

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fingerprinter import (
    unravel_index, dist_mat, electrostatic, electrostatic_sum,
    lennard_jones, lennard_jones_sum, fingerprints, Fingerprinter
)


class TestNumericalFunctions:
    """Test suite for numba-compiled numerical functions"""
    
    def test_unravel_index(self):
        """Test unravel_index function"""
        a, b = unravel_index(3, 4)
        assert a.shape == (12,)
        assert b.shape == (12,)
        assert np.array_equal(a[:4], [0, 0, 0, 0])
        assert np.array_equal(b[:4], [0, 1, 2, 3])
    
    def test_dist_mat(self):
        """Test distance matrix calculation"""
        xyz1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        xyz2 = np.array([[0, 0, 0], [0, 0, 1]])
        
        distances = dist_mat(xyz1, xyz2)
        
        assert distances.shape == (3, 2)
        assert np.isclose(distances[0, 0], 0.0)
        assert np.isclose(distances[0, 1], 1.0)
        assert np.isclose(distances[1, 0], 1.0)
        assert np.isclose(distances[2, 0], 1.0)
    
    def test_electrostatic(self):
        """Test electrostatic energy calculation"""
        # Test within cutoff
        energy = electrostatic(0.5, 1.0, -1.0)  # distance in nm, charges in e-
        assert energy != 0.0
        
        # Test beyond cutoff
        energy_far = electrostatic(1.5, 1.0, -1.0)
        assert energy_far == 0.0
    
    def test_electrostatic_sum(self):
        """Test sum of electrostatic interactions"""
        distances = np.array([[0.5, 0.8], [0.6, 1.2]])
        charges_i = np.array([1.0, -1.0])
        charges_j = np.array([1.0, -1.0])
        
        total_energy = electrostatic_sum(distances, charges_i, charges_j)
        assert isinstance(total_energy, float)
    
    def test_lennard_jones(self):
        """Test Lennard-Jones energy calculation"""
        # Test within cutoff
        energy = lennard_jones(0.5, 0.3, 0.3, 1.0, 1.0)
        assert energy != 0.0
        
        # Test beyond cutoff
        energy_far = lennard_jones(1.5, 0.3, 0.3, 1.0, 1.0)
        assert energy_far == 0.0
    
    def test_lennard_jones_sum(self):
        """Test sum of LJ interactions"""
        distances = np.array([[0.5, 0.8], [0.6, 1.2]])
        sigmas_i = np.array([0.3, 0.3])
        sigmas_j = np.array([0.3, 0.3])
        epsilons_i = np.array([1.0, 1.0])
        epsilons_j = np.array([1.0, 1.0])
        
        total_energy = lennard_jones_sum(distances, sigmas_i, sigmas_j, 
                                        epsilons_i, epsilons_j)
        assert isinstance(total_energy, float)
    
    def test_fingerprints(self):
        """Test fingerprint calculation"""
        # Setup test data
        xyzs = np.random.rand(100, 3)
        charges = np.random.rand(100)
        sigmas = np.ones(100) * 0.3
        epsilons = np.ones(100) * 1.0
        target_resmap = [list(range(10)), list(range(10, 20))]
        binder_inds = np.array(range(50, 80))
        
        lj_fp, es_fp = fingerprints(xyzs, charges, sigmas, epsilons,
                                   target_resmap, binder_inds)
        
        assert lj_fp.shape == (2,)
        assert es_fp.shape == (2,)


class TestFingerprinter:
    """Test suite for Fingerprinter class"""
    
    @patch('fingerprinter.AmberPrmtopFile')
    @patch('fingerprinter.mda.Universe')
    def test_init(self, mock_universe, mock_prmtop):
        """Test Fingerprinter initialization"""
        fp = Fingerprinter("test.prmtop", "test.dcd", 
                          target_selection="segid A",
                          binder_selection="segid B")
        
        assert fp.topology == Path("test.prmtop")
        assert fp.trajectory == Path("test.dcd")
        assert fp.target_selection == "segid A"
        assert fp.binder_selection == "segid B"
        assert fp.out == Path("fingerprint.npz")
    
    @patch('fingerprinter.AmberPrmtopFile')
    def test_assign_nonbonded_params(self, mock_prmtop):
        """Test assignment of non-bonded parameters"""
        # Create mock system
        mock_system = MagicMock()
        mock_system.getNumParticles.return_value = 10
        mock_system.getForces.return_value = [MagicMock()]
        
        mock_force = MagicMock()
        mock_force.getParticleParameters.return_value = (
            MagicMock(value=1.0, unit=MagicMock()),
            MagicMock(value=0.3, unit=MagicMock()),
            MagicMock(value=1.0, unit=MagicMock())
        )
        mock_system.getForces.return_value = [mock_force]
        
        mock_prmtop.return_value.createSystem.return_value = mock_system
        
        fp = Fingerprinter("test.prmtop")
        fp.assign_nonbonded_params()
        
        assert len(fp.charges) == 10
        assert len(fp.sigmas) == 10
        assert len(fp.epsilons) == 10
    
    @patch('fingerprinter.mda.Universe')
    def test_load_pdb(self, mock_universe):
        """Test loading PDB file"""
        fp = Fingerprinter("test.pdb")
        fp.load_pdb()
        
        mock_universe.assert_called_once_with(Path("test.pdb"))
        assert fp.u is not None
    
    @patch('fingerprinter.mda.Universe')
    def test_load_prmtop_with_trajectory(self, mock_universe):
        """Test loading prmtop with trajectory"""
        fp = Fingerprinter("test.prmtop", "test.dcd")
        fp.load_pdb()
        
        mock_universe.assert_called_once_with(Path("test.prmtop"), Path("test.dcd"))
    
    @patch('fingerprinter.mda.Universe')
    def test_assign_residue_mapping(self, mock_universe):
        """Test residue mapping assignment"""
        # Create mock universe with residues
        mock_u = MagicMock()
        mock_target = MagicMock()
        mock_binder = MagicMock()
        
        # Mock target residues
        mock_res1 = MagicMock()
        mock_res1.atoms.ix = np.array([0, 1, 2])
        mock_res2 = MagicMock()
        mock_res2.atoms.ix = np.array([3, 4, 5])
        mock_target.residues = [mock_res1, mock_res2]
        
        # Mock binder residues
        mock_res3 = MagicMock()
        mock_res3.atoms.ix = np.array([6, 7, 8])
        mock_binder.residues = [mock_res3]
        
        mock_u.select_atoms.side_effect = [mock_target, mock_binder]
        mock_universe.return_value = mock_u
        
        fp = Fingerprinter("test.pdb")
        fp.u = mock_u
        fp.assign_residue_mapping()
        
        assert len(fp.target_resmap) == 2
        assert len(fp.binder_resmap) == 1
        assert np.array_equal(fp.target_inds, np.array([0, 1, 2, 3, 4, 5]))
        assert np.array_equal(fp.binder_inds, np.array([6, 7, 8]))
    
    @patch('numpy.savez')
    def test_save(self, mock_savez):
        """Test saving fingerprint data"""
        fp = Fingerprinter("test.pdb")
        fp.target_fingerprint = np.random.rand(10, 5, 2)
        fp.binder_fingerprint = np.random.rand(10, 3, 2)
        
        fp.save()
        
        mock_savez.assert_called_once()
        args = mock_savez.call_args[0]
        assert str(args[0]) == "fingerprint.npz"
    
    def test_custom_output_path(self):
        """Test custom output path and name"""
        fp = Fingerprinter("test.pdb", out_path="/custom/path", 
                          out_name="custom_fingerprint.npz")
        
        assert fp.out == Path("/custom/path/custom_fingerprint.npz")


class TestIntegration:
    """Integration tests for the Fingerprinter"""
    
    def test_full_workflow_with_mock_data(self):
        """Test the complete fingerprinting workflow with mock data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This would require actual PDB/trajectory files
            # For unit testing, we'd mock all the MDAnalysis and OpenMM components
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
