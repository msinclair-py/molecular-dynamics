"""
Unit tests for analysis/interaction_energy.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestStaticInteractionEnergy:
    """Test suite for StaticInteractionEnergy class"""
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_static_interaction_energy_init(self, mock_platform):
        """Test StaticInteractionEnergy initialization"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        sie = StaticInteractionEnergy(
            pdb='test.pdb',
            chain='B',
            platform='CPU',
            first_residue=10,
            last_residue=50
        )
        
        assert sie.pdb == 'test.pdb'
        assert sie.chain == 'B'
        assert sie.first == 10
        assert sie.last == 50
        mock_platform.getPlatformByName.assert_called_with('CPU')
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_static_interaction_energy_init_defaults(self, mock_platform):
        """Test StaticInteractionEnergy default values"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        sie = StaticInteractionEnergy(pdb='test.pdb')
        
        assert sie.chain == 'A'
        assert sie.first is None
        assert sie.last is None
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_get_selection_full_chain(self, mock_platform):
        """Test get_selection for full chain"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        # Create mock topology
        mock_atom1 = MagicMock()
        mock_atom1.index = 0
        mock_atom1.residue.chain.id = 'A'
        
        mock_atom2 = MagicMock()
        mock_atom2.index = 1
        mock_atom2.residue.chain.id = 'A'
        
        mock_atom3 = MagicMock()
        mock_atom3.index = 2
        mock_atom3.residue.chain.id = 'B'
        
        mock_topology = MagicMock()
        mock_topology.atoms.return_value = [mock_atom1, mock_atom2, mock_atom3]
        
        sie = StaticInteractionEnergy(pdb='test.pdb', chain='A')
        sie.get_selection(mock_topology)
        
        # Should select only chain A atoms
        assert sie.selection == [0, 1]
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_get_selection_with_first_residue(self, mock_platform):
        """Test get_selection with first_residue filter"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_atoms = []
        for i in range(5):
            atom = MagicMock()
            atom.index = i
            atom.residue.chain.id = 'A'
            atom.residue.id = str(i + 1)  # Residues 1-5
            mock_atoms.append(atom)
        
        mock_topology = MagicMock()
        mock_topology.atoms.return_value = mock_atoms
        
        sie = StaticInteractionEnergy(pdb='test.pdb', chain='A', first_residue=3)
        sie.get_selection(mock_topology)
        
        # Should select residues >= 3 (indices 2, 3, 4)
        assert sie.selection == [2, 3, 4]
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_get_selection_with_last_residue(self, mock_platform):
        """Test get_selection with last_residue filter"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_atoms = []
        for i in range(5):
            atom = MagicMock()
            atom.index = i
            atom.residue.chain.id = 'A'
            atom.residue.id = str(i + 1)  # Residues 1-5
            mock_atoms.append(atom)
        
        mock_topology = MagicMock()
        mock_topology.atoms.return_value = mock_atoms
        
        sie = StaticInteractionEnergy(pdb='test.pdb', chain='A', last_residue=3)
        sie.get_selection(mock_topology)
        
        # Should select residues <= 3 (indices 0, 1, 2)
        assert sie.selection == [0, 1, 2]
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_get_selection_with_range(self, mock_platform):
        """Test get_selection with first and last residue"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_atoms = []
        for i in range(5):
            atom = MagicMock()
            atom.index = i
            atom.residue.chain.id = 'A'
            atom.residue.id = str(i + 1)  # Residues 1-5
            mock_atoms.append(atom)
        
        mock_topology = MagicMock()
        mock_topology.atoms.return_value = mock_atoms
        
        sie = StaticInteractionEnergy(
            pdb='test.pdb', chain='A', first_residue=2, last_residue=4
        )
        sie.get_selection(mock_topology)
        
        # Should select residues 2-4 (indices 1, 2, 3)
        assert sie.selection == [1, 2, 3]
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_interactions_property(self, mock_platform):
        """Test interactions property"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        sie = StaticInteractionEnergy(pdb='test.pdb')
        sie.lj = -5.0
        sie.coulomb = -10.0
        
        result = sie.interactions
        
        assert result.shape == (2, 1)
        assert result[0, 0] == -5.0
        assert result[1, 0] == -10.0
    
    def test_energy_static_method(self):
        """Test energy static method"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_context = MagicMock()
        mock_energy = MagicMock()
        mock_energy.getPotentialEnergy.return_value = 100.0
        mock_context.getState.return_value = mock_energy
        
        result = StaticInteractionEnergy.energy(
            mock_context,
            solute_coulomb_scale=1,
            solute_lj_scale=0,
            solvent_coulomb_scale=1,
            solvent_lj_scale=0
        )
        
        mock_context.setParameter.assert_any_call("solute_coulomb_scale", 1)
        mock_context.setParameter.assert_any_call("solute_lj_scale", 0)
        assert result == 100.0
    
    @patch('molecular_simulations.analysis.interaction_energy.PDBFixer')
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_fix_pdb(self, mock_platform, mock_pdbfixer):
        """Test fix_pdb method"""
        from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_fixer = MagicMock()
        mock_fixer.positions = [[0, 0, 0]]
        mock_fixer.topology = MagicMock()
        mock_pdbfixer.return_value = mock_fixer
        
        sie = StaticInteractionEnergy(pdb='test.pdb')
        positions, topology = sie.fix_pdb()
        
        mock_fixer.findMissingResidues.assert_called_once()
        mock_fixer.findMissingAtoms.assert_called_once()
        mock_fixer.addMissingAtoms.assert_called_once()
        mock_fixer.addMissingHydrogens.assert_called_once_with(7.0)


class TestInteractionEnergyFrame:
    """Test suite for InteractionEnergyFrame class"""
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_interaction_energy_frame_init(self, mock_platform):
        """Test InteractionEnergyFrame initialization"""
        from molecular_simulations.analysis.interaction_energy import InteractionEnergyFrame
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_system = MagicMock()
        mock_topology = MagicMock()
        
        ief = InteractionEnergyFrame(
            system=mock_system,
            top=mock_topology,
            chain='C',
            platform='CPU',
            first_residue=5,
            last_residue=15
        )
        
        assert ief.system == mock_system
        assert ief.top == mock_topology
        assert ief.chain == 'C'
        assert ief.first == 5
        assert ief.last == 15
    
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_interaction_energy_frame_get_system(self, mock_platform):
        """Test InteractionEnergyFrame get_system method"""
        from molecular_simulations.analysis.interaction_energy import InteractionEnergyFrame
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_system = MagicMock()
        mock_topology = MagicMock()
        mock_topology.atoms.return_value = []
        
        ief = InteractionEnergyFrame(
            system=mock_system,
            top=mock_topology
        )
        
        result = ief.get_system()
        
        assert result == mock_system
        assert hasattr(ief, 'selection')


class TestDynamicInteractionEnergy:
    """Test suite for DynamicInteractionEnergy class"""
    
    @patch('molecular_simulations.analysis.interaction_energy.md')
    @patch('molecular_simulations.analysis.interaction_energy.AmberPrmtopFile')
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_dynamic_interaction_energy_init(self, mock_platform, mock_prmtop, mock_md):
        """Test DynamicInteractionEnergy initialization"""
        from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_prmtop_instance = MagicMock()
        mock_prmtop_instance.createSystem.return_value = MagicMock()
        mock_prmtop.return_value = mock_prmtop_instance
        
        mock_traj = MagicMock()
        mock_traj.xyz = np.zeros((10, 100, 3))
        mock_md.load.return_value = mock_traj
        
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / 'system.prmtop'
            traj_path = Path(tmpdir) / 'traj.dcd'
            top_path.write_text("dummy")
            traj_path.write_text("dummy")
            
            die = DynamicInteractionEnergy(
                top=top_path,
                traj=traj_path,
                stride=2,
                chain='A',
                platform='CPU',
                first_residue=1,
                last_residue=10,
                progress_bar=True
            )
            
            assert die.stride == 2
            assert die.progress is True
            assert die.coordinates.shape == (10, 100, 3)
    
    @patch('molecular_simulations.analysis.interaction_energy.md')
    @patch('molecular_simulations.analysis.interaction_energy.PDBFile')
    @patch('molecular_simulations.analysis.interaction_energy.ForceField')
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_build_system_pdb(self, mock_platform, mock_ff, mock_pdb, mock_md):
        """Test build_system with PDB file"""
        from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_pdb_instance = MagicMock()
        mock_pdb_instance.topology = MagicMock()
        mock_pdb.return_value = mock_pdb_instance
        
        mock_ff_instance = MagicMock()
        mock_ff_instance.createSystem.return_value = MagicMock()
        mock_ff.return_value = mock_ff_instance
        
        mock_traj = MagicMock()
        mock_traj.xyz = np.zeros((5, 50, 3))
        mock_md.load.return_value = mock_traj
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / 'system.pdb'
            traj_path = Path(tmpdir) / 'traj.dcd'
            pdb_path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            traj_path.write_text("dummy")
            
            die = DynamicInteractionEnergy(
                top=pdb_path,
                traj=traj_path
            )
            
            # Should have built system from PDB
            mock_ff.assert_called()
    
    @patch('molecular_simulations.analysis.interaction_energy.md')
    @patch('molecular_simulations.analysis.interaction_energy.AmberPrmtopFile')
    @patch('molecular_simulations.analysis.interaction_energy.Platform')
    def test_build_system_unsupported(self, mock_platform, mock_prmtop, mock_md):
        """Test build_system with unsupported topology"""
        from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy
        
        mock_platform.getPlatformByName.return_value = MagicMock()
        
        mock_traj = MagicMock()
        mock_traj.xyz = np.zeros((5, 50, 3))
        mock_md.load.return_value = mock_traj
        
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / 'system.xyz'  # Unsupported
            traj_path = Path(tmpdir) / 'traj.dcd'
            top_path.write_text("dummy")
            traj_path.write_text("dummy")
            
            with pytest.raises(NotImplementedError):
                DynamicInteractionEnergy(top=top_path, traj=traj_path)


class TestInteractionEnergyAbstract:
    """Test the abstract base class"""

    def test_abstract_class_cannot_instantiate(self):
        """Test that InteractionEnergy cannot be instantiated"""
        from molecular_simulations.analysis.interaction_energy import InteractionEnergy

        with pytest.raises(TypeError):
            InteractionEnergy()


class TestDynamicInteractionEnergyAdditional:
    """Additional tests for DynamicInteractionEnergy class."""

    def test_setup_pbar(self):
        """Test setup_pbar creates progress bar."""
        with patch('molecular_simulations.analysis.interaction_energy.Platform') as mock_platform, \
             patch('molecular_simulations.analysis.interaction_energy.tqdm') as mock_tqdm:
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.zeros((100, 500, 3))

            die.setup_pbar()

            mock_tqdm.assert_called_once_with(total=100, position=0, leave=False)

    def test_compute_energies_no_progress_bar(self):
        """Test compute_energies without progress bar."""
        with patch('molecular_simulations.analysis.interaction_energy.Platform') as mock_platform:
            mock_platform.getPlatformByName.return_value = MagicMock()

            from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.zeros((10, 500, 3))
            die.stride = 2
            die.progress = False

            mock_ie = MagicMock()
            mock_ie.lj = -10.0
            mock_ie.coulomb = -20.0
            die.IE = mock_ie

            die.compute_energies()

            # With stride=2 and 10 frames, should compute 5 energies
            assert die.energies.shape == (5, 2)
            assert mock_ie.compute.call_count == 5
            # Check that energies were filled
            assert np.all(die.energies[:, 0] == -10.0)
            assert np.all(die.energies[:, 1] == -20.0)

    def test_compute_energies_with_progress_bar(self):
        """Test compute_energies with progress bar enabled."""
        with patch('molecular_simulations.analysis.interaction_energy.Platform') as mock_platform, \
             patch('molecular_simulations.analysis.interaction_energy.tqdm') as mock_tqdm:
            mock_platform.getPlatformByName.return_value = MagicMock()
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)
            die.coordinates = np.zeros((10, 500, 3))
            die.stride = 1
            die.progress = True

            mock_ie = MagicMock()
            mock_ie.lj = -5.0
            mock_ie.coulomb = -15.0
            die.IE = mock_ie

            die.compute_energies()

            assert die.energies.shape == (10, 2)
            assert mock_pbar.update.call_count == 10
            mock_pbar.close.assert_called_once()

    def test_load_traj(self):
        """Test load_traj loads trajectory with mdtraj."""
        with patch('molecular_simulations.analysis.interaction_energy.Platform') as mock_platform, \
             patch('molecular_simulations.analysis.interaction_energy.md') as mock_md:
            mock_platform.getPlatformByName.return_value = MagicMock()
            mock_traj = MagicMock()
            mock_traj.xyz = np.random.rand(100, 500, 3)
            mock_md.load.return_value = mock_traj

            from molecular_simulations.analysis.interaction_energy import DynamicInteractionEnergy

            die = DynamicInteractionEnergy.__new__(DynamicInteractionEnergy)

            result = die.load_traj(Path('top.pdb'), Path('traj.dcd'))

            mock_md.load.assert_called_once_with('traj.dcd', top='top.pdb')
            assert np.array_equal(result, mock_traj.xyz)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
