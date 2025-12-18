"""
Unit tests for simulate/multires_simulator.py module

Note: These tests mock calvados and cg2all dependencies at the sys.modules level
since these are optional/difficult dependencies.
"""
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


# Create comprehensive mocks for all difficult dependencies
@pytest.fixture(autouse=True)
def mock_difficult_dependencies():
    """Mock calvados, cg2all, and related dependencies before any imports"""
    
    # Mock calvados
    mock_calvados = MagicMock()
    mock_calvados.sim = MagicMock()
    mock_calvados_cfg = MagicMock()
    mock_calvados_cfg.Config = MagicMock()
    mock_calvados_cfg.Job = MagicMock()
    mock_calvados_cfg.Components = MagicMock()
    mock_calvados.cfg = mock_calvados_cfg
    
    # Mock cg2all
    mock_cg2all = MagicMock()
    mock_cg2all_script = MagicMock()
    mock_cg2all_convert = MagicMock()
    mock_cg2all_script.convert_cg2all = mock_cg2all_convert
    mock_cg2all_convert.main = MagicMock()
    
    # Mock parmed
    mock_parmed = MagicMock()
    mock_parmed.openmm = MagicMock()
    mock_parmed.openmm.load_topology = MagicMock()
    
    # Insert mocks into sys.modules
    with patch.dict(sys.modules, {
        'calvados': mock_calvados,
        'calvados.cfg': mock_calvados_cfg,
        'calvados.sim': mock_calvados.sim,
        'cg2all': mock_cg2all,
        'cg2all.script': mock_cg2all_script,
        'cg2all.script.convert_cg2all': mock_cg2all_convert,
        'parmed': mock_parmed,
    }):
        yield {
            'calvados': mock_calvados,
            'cg2all': mock_cg2all,
            'parmed': mock_parmed,
        }


class TestSanderMinDefaults:
    """Test suite for sander_min_defaults dataclass"""
    
    def test_sander_min_defaults_values(self, mock_difficult_dependencies):
        """Test default values for sander minimization"""
        from molecular_simulations.simulate.multires_simulator import sander_min_defaults
        
        defaults = sander_min_defaults()
        
        assert defaults.imin == 1
        assert defaults.maxcyc == 5000
        assert defaults.ncyc == 2500
        assert defaults.ntb == 0
        assert defaults.ntr == 0
        assert defaults.cut == 10.0
        assert defaults.ntpr == 10000
        assert defaults.ntwr == 5000
        assert defaults.ntxo == 1
    
    def test_sander_min_defaults_mdin_contents(self, mock_difficult_dependencies):
        """Test that mdin_contents is properly formatted"""
        from molecular_simulations.simulate.multires_simulator import sander_min_defaults
        
        defaults = sander_min_defaults()
        
        assert 'Minimization input' in defaults.mdin_contents
        assert 'imin=1' in defaults.mdin_contents
        assert 'maxcyc=5000' in defaults.mdin_contents
        assert 'ncyc=2500' in defaults.mdin_contents
        assert 'ntb=0' in defaults.mdin_contents
        assert 'cut=10.0' in defaults.mdin_contents
        assert '&cntrl' in defaults.mdin_contents


class TestSanderMinimize:
    """Test suite for sander_minimize function"""
    
    def test_sander_minimize_success(self, mock_difficult_dependencies):
        """Test sander_minimize runs successfully"""
        from molecular_simulations.simulate.multires_simulator import sander_minimize
        
        with patch('molecular_simulations.simulate.multires_simulator.subprocess') as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(returncode=0)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                
                # Create dummy files
                (tmpdir / 'system.inpcrd').write_text("coords")
                (tmpdir / 'system.prmtop').write_text("topology")
                
                sander_minimize(
                    path=tmpdir,
                    inpcrd_file='system.inpcrd',
                    prmtop_file='system.prmtop',
                    sander_cmd='sander'
                )
                
                # Should have called subprocess.run
                mock_subprocess.run.assert_called_once()
    
    def test_sander_minimize_failure(self, mock_difficult_dependencies):
        """Test sander_minimize raises error on failure"""
        from molecular_simulations.simulate.multires_simulator import sander_minimize
        
        with patch('molecular_simulations.simulate.multires_simulator.subprocess') as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(
                returncode=1,
                stderr="Error message",
                stdout="Output"
            )
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                
                (tmpdir / 'system.inpcrd').write_text("coords")
                (tmpdir / 'system.prmtop').write_text("topology")
                
                with pytest.raises(RuntimeError, match="sander error"):
                    sander_minimize(
                        path=tmpdir,
                        inpcrd_file='system.inpcrd',
                        prmtop_file='system.prmtop',
                        sander_cmd='sander'
                    )


class TestMultiResolutionSimulator:
    """Test suite for MultiResolutionSimulator class"""
    
    def test_multires_init(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator initialization"""
        from molecular_simulations.simulate.multires_simulator import MultiResolutionSimulator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            cg_params = {'config': {}, 'components': {}}
            aa_params = {
                'solvation_scheme': 'implicit',
                'protein': True,
                'rna': False,
                'dna': False,
                'phos_protein': False,
                'use_amber': True,
                'out': 'system.pdb',
                'equilibration_steps': 1000,
                'production_steps': 10000,
                'device_ids': [0],
            }
            
            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=3,
                cg_params=cg_params,
                aa_params=aa_params,
                cg2all_bin='convert_cg2all',
                cg2all_ckpt=None,
                amberhome='/fake/amber'
            )
            
            assert sim.path == tmpdir
            assert sim.input_pdb == 'protein.pdb'
            assert sim.n_rounds == 3
            assert sim.amberhome == Path('/fake/amber')
    
    def test_multires_init_no_amberhome(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator initialization without amberhome"""
        from molecular_simulations.simulate.multires_simulator import MultiResolutionSimulator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / 'protein.pdb'
            pdb_path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            
            sim = MultiResolutionSimulator(
                path=tmpdir,
                input_pdb='protein.pdb',
                n_rounds=1,
                cg_params={},
                aa_params={},
                amberhome=None
            )
            
            assert sim.amberhome is None
    
    def test_multires_from_toml(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator.from_toml classmethod"""
        from molecular_simulations.simulate.multires_simulator import MultiResolutionSimulator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create TOML config
            toml_content = f"""
[settings]
path = "{tmpdir}"
input_pdb = "protein.pdb"
n_rounds = 2
cg2all_bin = "convert_cg2all"
cg2all_ckpt = "/path/to/ckpt"
amberhome = "/fake/amber"

[[cg_params]]
test = "value"

[aa_params]
solvation_scheme = "explicit"
"""
            config_path = tmpdir / 'config.toml'
            config_path.write_text(toml_content)
            
            sim = MultiResolutionSimulator.from_toml(config_path)
            
            assert sim.n_rounds == 2
            assert sim.cg2all_bin == 'convert_cg2all'
            assert sim.cg2all_ckpt == '/path/to/ckpt'
    
    def test_multires_from_toml_minimal(self, mock_difficult_dependencies):
        """Test MultiResolutionSimulator.from_toml with minimal config"""
        from molecular_simulations.simulate.multires_simulator import MultiResolutionSimulator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Minimal TOML config (no optional fields)
            toml_content = f"""
[settings]
path = "{tmpdir}"
input_pdb = "protein.pdb"
n_rounds = 1

[[cg_params]]
test = "value"

[aa_params]
solvation_scheme = "implicit"
"""
            config_path = tmpdir / 'config.toml'
            config_path.write_text(toml_content)
            
            sim = MultiResolutionSimulator.from_toml(config_path)
            
            assert sim.n_rounds == 1
            assert sim.cg2all_bin == 'convert_cg2all'  # Default
            assert sim.cg2all_ckpt is None  # Default
            assert sim.amberhome is None  # Default
    
    def test_strip_solvent(self, mock_difficult_dependencies):
        """Test strip_solvent static method"""
        from molecular_simulations.simulate.multires_simulator import MultiResolutionSimulator
        
        # Get the parmed mock from our fixture
        mock_parmed = mock_difficult_dependencies['parmed']
        mock_struc = MagicMock()
        mock_parmed.openmm.load_topology.return_value = mock_struc
        
        mock_simulation = MagicMock()
        mock_simulation.topology = MagicMock()
        mock_simulation.system = MagicMock()
        mock_simulation.context.getState.return_value.getPositions.return_value = [[0, 0, 0]]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'protein.pdb'
            
            MultiResolutionSimulator.strip_solvent(
                mock_simulation,
                output_pdb=str(output_path)
            )
            
            # Should have called strip with solvent residues
            mock_struc.strip.assert_called_once()
            mock_struc.save.assert_called_once_with(str(output_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
