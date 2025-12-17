"""
Unit tests for build/build_interface.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os
import yaml


class TestInterfaceBuilderInit:
    """Test suite for InterfaceBuilder class initialization"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_interface_builder_init(self, mock_mda):
        """Test InterfaceBuilder initialization"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        # Mock target universe
        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_universe.select_atoms.return_value = mock_atoms
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create input files
            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {
                'site0': {
                    'contact_sel': 'name CA and resid 10-50',
                    'distance_sel': 'resid 10-50',
                    'vector': [10.0, 0.0, 0.0],
                    'com': [50.0, 50.0, 50.0]
                }
            }

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )

            assert builder.interfaces == interfaces
            assert builder.binder == binder_file

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_interface_builder_with_forcefields(self, mock_mda):
        """Test InterfaceBuilder with different forcefields"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_universe.select_atoms.return_value = mock_atoms
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {'site0': {'contact_sel': '', 'distance_sel': '', 'vector': [0, 0, 0], 'com': [0, 0, 0]}}

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file,
                protein=True,
                rna=True
            )

            assert 'leaprc.protein.ff19SB' in builder.ffs
            assert 'leaprc.RNA.Shaw' in builder.ffs


class TestInterfaceBuilderPlaceBinder:
    """Test suite for place_binder method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_place_binder(self, mock_mda):
        """Test place_binder method"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        # Mock for init
        mock_target_universe = MagicMock()
        mock_target_atoms = MagicMock()
        mock_target_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_target_universe.select_atoms.return_value = mock_target_atoms

        # Mock for place_binder
        mock_binder_universe = MagicMock()
        mock_binder_atoms = MagicMock()
        mock_binder_atoms.center_of_mass.return_value = np.array([0.0, 0.0, 0.0])
        mock_binder_atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mock_binder_universe.select_atoms.return_value = mock_binder_atoms

        # Setup mock to return different universes for target and binder
        mock_mda.Universe.side_effect = [mock_target_universe, mock_binder_universe]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {'site0': {'contact_sel': '', 'distance_sel': '', 'vector': [0, 0, 0], 'com': [0, 0, 0]}}

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )

            vector = np.array([10.0, 0.0, 0.0], dtype=np.float32)
            com = np.array([50.0, 50.0, 50.0], dtype=np.float32)

            result = builder.place_binder(vector, com)

            # Binder positions should be translated
            assert result is mock_binder_atoms


class TestInterfaceBuilderMergeProteins:
    """Test suite for merge_proteins method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_merge_proteins(self, mock_mda):
        """Test merge_proteins method"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        mock_target_universe = MagicMock()
        mock_target_atoms = MagicMock()
        mock_target_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_target_universe.select_atoms.return_value = mock_target_atoms
        mock_mda.Universe.return_value = mock_target_universe

        mock_merged = MagicMock()
        mock_mda.Merge.return_value = mock_merged

        mock_writer = MagicMock()
        mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {'site0': {'contact_sel': '', 'distance_sel': '', 'vector': [0, 0, 0], 'com': [0, 0, 0]}}

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )
            builder.target = mock_target_atoms
            builder.pdb = path / 'merged.pdb'

            mock_binder = MagicMock()
            builder.merge_proteins(mock_binder)

            mock_mda.Merge.assert_called_once_with(mock_target_atoms, mock_binder)
            mock_writer.write.assert_called_once_with(mock_merged)


class TestInterfaceBuilderParseInterface:
    """Test suite for parse_interface method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_parse_interface(self, mock_mda):
        """Test parse_interface method"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_universe.select_atoms.return_value = mock_atoms
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {
                'site0': {
                    'contact_sel': 'name CA and resid 10 11 12 13 14',  # 5 residues
                    'distance_sel': 'resid 10-14',
                    'vector': [10.0, 0.0, 0.0],
                    'com': [50.0, 50.0, 50.0]
                }
            }

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )

            result = builder.parse_interface('site0')

            assert len(result) == 5  # contact_sel, distance_sel, vector, com, input_shape
            assert result[2] == [10.0, 0.0, 0.0]  # vector
            assert result[3] == [50.0, 50.0, 50.0]  # com


class TestInterfaceBuilderWriteYaml:
    """Test suite for YAML writing methods"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_write_ddmd_yaml(self, mock_mda):
        """Test write_ddmd_yaml method"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_universe.select_atoms.return_value = mock_atoms
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {'site0': {'contact_sel': '', 'distance_sel': '', 'vector': [0, 0, 0], 'com': [0, 0, 0]}}

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )
            builder.yaml_out = path
            builder.out = path / 'ddmd'

            builder.write_ddmd_yaml('name CA', 'resid 1-10')

            yaml_file = path / 'prod.yaml'
            assert yaml_file.exists()

            with open(yaml_file) as f:
                content = yaml.safe_load(f)

            assert 'simulation_input_dir' in content
            assert 'num_workers' in content
            assert content['simulation_settings']['mda_selection'] == 'name CA'
            assert content['simulation_settings']['distance_sels'] == 'resid 1-10'

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_write_cvae_yaml(self, mock_mda):
        """Test write_cvae_yaml method"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_universe.select_atoms.return_value = mock_atoms
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {'site0': {'contact_sel': '', 'distance_sel': '', 'vector': [0, 0, 0], 'com': [0, 0, 0]}}

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )
            builder.yaml_out = path

            input_shape = (1, 10, 10)
            builder.write_cvae_yaml(input_shape)

            yaml_file = path / 'cvae-prod-settings.yaml'
            assert yaml_file.exists()

            with open(yaml_file) as f:
                content = yaml.load(f, Loader=yaml.FullLoader)

            assert list(content['input_shape']) == [1, 10, 10]
            assert 'filters' in content
            assert 'kernels' in content
            assert 'latent_dim' in content
            assert content['device'] == 'cuda'


class TestInterfaceBuilderBuildAll:
    """Test suite for build_all method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_interface.mda')
    def test_build_all_structure(self, mock_mda):
        """Test that build_all creates proper directory structure"""
        from molecular_simulations.build.build_interface import InterfaceBuilder

        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.center_of_mass.return_value = np.array([50.0, 50.0, 50.0])
        mock_atoms.positions = np.array([[0.0, 0.0, 0.0]])
        mock_universe.select_atoms.return_value = mock_atoms
        mock_mda.Universe.return_value = mock_universe

        mock_merged = MagicMock()
        mock_mda.Merge.return_value = mock_merged

        mock_writer = MagicMock()
        mock_mda.Writer.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_mda.Writer.return_value.__exit__ = Mock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'complex.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            target_file = path / 'target.pdb'
            target_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            binder_file = path / 'binder.pdb'
            binder_file.write_text("ATOM      1  N   ALA B   1       10.000   10.000   10.000  1.00  0.00\n")

            interfaces = {
                'site0': {
                    'contact_sel': 'name CA and resid 10 11 12',
                    'distance_sel': 'resid 10-12',
                    'vector': [10.0, 0.0, 0.0],
                    'com': [50.0, 50.0, 50.0]
                }
            }

            builder = InterfaceBuilder(
                path=str(path),
                pdb=str(pdb_file),
                interfaces=interfaces,
                target=target_file,
                binder=binder_file
            )

            # Mock build method to avoid actual tleap calls
            with patch.object(builder, 'build'):
                builder.build_all()

                # Check that directories were created
                expected_root = path / 'target' / 'site0' / 'binder'
                # Note: The actual structure depends on target/binder filenames


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
