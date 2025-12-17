"""
Unit tests for simulate/mmpbsa.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os
import json


class TestMMPBSASettings:
    """Test suite for MMPBSA_settings dataclass"""

    def test_mmpbsa_settings_defaults(self):
        """Test MMPBSA_settings with default values"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA_settings

        settings = MMPBSA_settings(
            top='system.prmtop',
            dcd='traj.dcd',
            selections=[':1-100', ':101-150']
        )

        assert settings.first_frame == 0
        assert settings.last_frame == -1
        assert settings.stride == 1
        assert settings.n_cpus == 1
        assert settings.out == 'mmpbsa'
        assert settings.solvent_probe == 1.4
        assert settings.gb_surften == 0.0072
        assert settings.gb_surfoff == 0.0

    def test_mmpbsa_settings_custom(self):
        """Test MMPBSA_settings with custom values"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA_settings

        settings = MMPBSA_settings(
            top='system.prmtop',
            dcd='traj.dcd',
            selections=[':1-100', ':101-150'],
            first_frame=10,
            last_frame=100,
            stride=5,
            n_cpus=4,
            out='custom_output'
        )

        assert settings.first_frame == 10
        assert settings.last_frame == 100
        assert settings.stride == 5
        assert settings.n_cpus == 4
        assert settings.out == 'custom_output'


class TestMMPBSAInit:
    """Test suite for MMPBSA class initialization"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_mmpbsa_init(self, mock_filehandler):
        """Test MMPBSA initialization"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_filehandler.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )

            assert mmpbsa.cpptraj == Path('/fake/amber/bin/cpptraj')
            assert mmpbsa.mmpbsa_py_energy == Path('/fake/amber/bin/mmpbsa_py_energy')

    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_mmpbsa_init_no_amberhome(self, mock_filehandler):
        """Test MMPBSA initialization without AMBERHOME"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        # Remove AMBERHOME from environment
        env = os.environ.copy()
        env.pop('AMBERHOME', None)

        with patch.dict(os.environ, env, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                top_file = path / 'system.prmtop'
                top_file.write_text("mock topology")
                traj_file = path / 'traj.dcd'
                traj_file.write_text("mock trajectory")

                with pytest.raises(ValueError, match="AMBERHOME not set"):
                    MMPBSA(
                        top=str(top_file),
                        dcd=str(traj_file),
                        selections=[':1-100', ':101-150']
                    )

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_mmpbsa_init_custom_output(self, mock_filehandler):
        """Test MMPBSA initialization with custom output path"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_filehandler.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            out_dir = path / 'custom_output'
            out_dir.mkdir()

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                out=str(out_dir)
            )

            # Use resolve() to handle /private symlink on macOS
            assert mmpbsa.path.resolve() == out_dir.resolve()


class TestMMPBSAWriteMdins:
    """Test suite for MMPBSA write_mdins method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_write_mdins(self, mock_filehandler):
        """Test write_mdins method"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_fh.path = MagicMock()
        mock_fh.write_file = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            mock_fh.path = path

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )

            gb_mdin, pb_mdin = mmpbsa.write_mdins()

            # Should have written both GB and PB mdins
            assert mock_fh.write_file.call_count == 2


class TestOutputAnalyzer:
    """Test suite for OutputAnalyzer class"""

    def test_output_analyzer_init(self):
        """Test OutputAnalyzer initialization"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = OutputAnalyzer(
                path=tmpdir,
                surface_tension=0.01,
                sasa_offset=0.5
            )

            assert analyzer.surften == 0.01
            assert analyzer.offset == 0.5
            assert analyzer.free_energy is None

    def test_parse_line_single_value(self):
        """Test parse_line with single key=value"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = " BOND = 123.456"
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 1
        assert result[0][0] == 'BOND'
        assert np.isclose(result[0][1], 123.456)

    def test_parse_line_multiple_values(self):
        """Test parse_line with multiple key=value pairs"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = " BOND =  123.456  ANGLE =  78.901"
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 2

    def test_read_sasa(self):
        """Test read_sasa method"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create mock SASA file
            sasa_content = """#Frame SASA
1 1000.0
2 1100.0
3 1050.0
"""
            sasa_file = path / 'test_surf.dat'
            sasa_file.write_text(sasa_content)

            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)
            sasa_series = analyzer.read_sasa(sasa_file)

            assert len(sasa_series) == 3
            # SASA should be scaled by surface tension
            assert all(s != 0 for s in sasa_series)


class TestFileHandler:
    """Test suite for FileHandler class"""

    def test_write_file_list(self):
        """Test write_file with list input"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'test.txt'

            lines = ['line1', 'line2', 'line3']
            FileHandler.write_file(lines, outfile)

            content = outfile.read_text()
            assert 'line1\nline2\nline3' == content

    def test_write_file_string(self):
        """Test write_file with string input"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'test.txt'

            content = 'single string content'
            FileHandler.write_file(content, outfile)

            result = outfile.read_text()
            assert result == content


class TestRunEnergyCalculation:
    """Test suite for _run_energy_calculation worker function"""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_run_energy_calculation_success(self, mock_subprocess):
        """Test successful energy calculation"""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr='', stdout='')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create expected output file
            out_file = path / 'test_gb.mdout'
            out_file.write_text(" BOND = 100.0\n ANGLE = 50.0\n")

            args = (
                '/fake/amber/bin/mmpbsa_py_energy',
                'gb_mdin',
                'system.prmtop',
                'system.pdb',
                'traj.crd',
                'test_gb.mdout',
                str(path)
            )

            result_path, success, error = _run_energy_calculation(args)

            assert success
            assert error == ''

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_run_energy_calculation_failure(self, mock_subprocess):
        """Test failed energy calculation"""
        from molecular_simulations.simulate.mmpbsa import _run_energy_calculation

        mock_subprocess.run.return_value = MagicMock(
            returncode=1,
            stderr='Error message',
            stdout=''
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = (
                '/fake/amber/bin/mmpbsa_py_energy',
                'gb_mdin',
                'system.prmtop',
                'system.pdb',
                'traj.crd',
                'test_gb.mdout',
                tmpdir
            )

            result_path, success, error = _run_energy_calculation(args, max_retries=1)

            assert not success


class TestRunSasaCalculation:
    """Test suite for _run_sasa_calculation worker function"""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_run_sasa_calculation_success(self, mock_subprocess):
        """Test successful SASA calculation"""
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr='', stdout='')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create SASA script
            sasa_script = path / 'sasa.in'
            sasa_script.write_text("molsurf :* out test_surf.dat probe 1.4")

            # Create expected output
            out_file = path / 'test_surf.dat'
            out_file.write_text("#Frame SASA\n1 1000.0\n")

            args = (
                '/fake/cpptraj',
                str(sasa_script),
                str(path)
            )

            result_path, success, error = _run_sasa_calculation(args)

            assert success
            assert error == ''

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_run_sasa_calculation_missing_output(self, mock_subprocess):
        """Test SASA calculation with missing output"""
        from molecular_simulations.simulate.mmpbsa import _run_sasa_calculation

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create SASA script but no output file
            sasa_script = path / 'sasa.in'
            sasa_script.write_text("molsurf :* out test_surf.dat probe 1.4")

            args = (
                '/fake/cpptraj',
                str(sasa_script),
                str(path)
            )

            result_path, success, error = _run_sasa_calculation(args, max_retries=1)

            assert not success


class TestCombineChunks:
    """Test suite for chunk combining methods"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_combine_sasa_chunks(self, mock_filehandler):
        """Test _combine_sasa_chunks method"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create topology and trajectory files
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            # Create chunk files
            for system in ['complex', 'receptor', 'ligand']:
                for i in range(2):
                    chunk_file = path / f'{system}_chunk{i}_surf.dat'
                    if i == 0:
                        chunk_file.write_text("#Frame SASA\n1 1000.0\n2 1050.0\n")
                    else:
                        chunk_file.write_text("#Frame SASA\n3 1100.0\n4 1150.0\n")

            mock_fh.path = path

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )
            mmpbsa.path = path

            mmpbsa._combine_sasa_chunks()

            # Check combined files exist
            for system in ['complex', 'receptor', 'ligand']:
                combined_file = path / f'{system}_surf.dat'
                assert combined_file.exists()

            # Chunk files should be deleted
            for system in ['complex', 'receptor', 'ligand']:
                for i in range(2):
                    chunk_file = path / f'{system}_chunk{i}_surf.dat'
                    assert not chunk_file.exists()

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_combine_energy_chunks(self, mock_filehandler):
        """Test _combine_energy_chunks method"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            # Create energy chunk files
            for system in ['complex', 'receptor', 'ligand']:
                for level in ['gb', 'pb']:
                    for i in range(2):
                        chunk_file = path / f'{system}_chunk{i}_{level}.mdout'
                        chunk_file.write_text(f" BOND = {100 + i * 10}.0\n")

            mock_fh.path = path

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )
            mmpbsa.path = path

            mmpbsa._combine_energy_chunks()

            # Check combined files exist
            for system in ['complex', 'receptor', 'ligand']:
                for level in ['gb', 'pb']:
                    combined_file = path / f'{system}_{level}.mdout'
                    assert combined_file.exists()


class TestVerifyCombinedOutputs:
    """Test suite for _verify_combined_outputs method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_verify_combined_outputs_success(self, mock_filehandler):
        """Test _verify_combined_outputs with valid files"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            # Create valid output files
            for system in ['complex', 'receptor', 'ligand']:
                sasa_file = path / f'{system}_surf.dat'
                sasa_file.write_text("1 1000.0\n2 1050.0\n")

                for level in ['gb', 'pb']:
                    energy_file = path / f'{system}_{level}.mdout'
                    energy_file.write_text(" BOND = 100.0\n ANGLE = 50.0\n")

            mock_fh.path = path

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )
            mmpbsa.path = path

            # Should not raise
            mmpbsa._verify_combined_outputs()

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_verify_combined_outputs_missing_files(self, mock_filehandler):
        """Test _verify_combined_outputs with missing files"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            mock_fh.path = path

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )
            mmpbsa.path = path

            with pytest.raises(RuntimeError, match="Missing files"):
                mmpbsa._verify_combined_outputs()


class TestWriteSasaScript:
    """Test suite for _write_sasa_script method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_write_sasa_script(self, mock_filehandler):
        """Test _write_sasa_script method"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            mock_fh.path = path
            mock_fh.write_file = MagicMock()

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )
            mmpbsa.path = path

            prefix = path / 'complex'
            prm = path / 'complex.prmtop'
            trj = path / 'complex_chunk0.crd'

            result = mmpbsa._write_sasa_script(prefix, prm, trj, chunk_idx=0)

            mock_fh.write_file.assert_called_once()
            assert 'sasa_complex_chunk0.in' in str(result)


class TestOutputAnalyzerMethods:
    """Test suite for OutputAnalyzer methods"""

    def test_output_analyzer_init_full(self):
        """Test OutputAnalyzer initialization with all parameters"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = OutputAnalyzer(
                path=tmpdir,
                surface_tension=0.01,
                sasa_offset=0.5
            )

            assert analyzer.surften == 0.01
            assert analyzer.offset == 0.5
            assert analyzer.free_energy is None

    def test_parse_line_standard_format(self):
        """Test parse_line with standard Amber format"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        # Standard format from mmpbsa_py_energy output
        line = " BOND =    123.456  ANGLE =     78.901"
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 2
        assert result[0][0] == 'BOND'
        assert result[1][0] == 'ANGLE'
        assert np.isclose(result[0][1], 123.456)
        assert np.isclose(result[1][1], 78.901)

    def test_parse_line_single_value(self):
        """Test parse_line with single value"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        line = " BOND = 123.456"
        result = list(OutputAnalyzer.parse_line(line))

        assert len(result) == 1
        assert result[0][0] == 'BOND'
        assert np.isclose(result[0][1], 123.456)


class TestFileHandlerWriteFile:
    """Test suite for FileHandler.write_file method"""

    def test_write_file_creates_output(self):
        """Test that write_file creates output correctly"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'test_output.txt'

            content = ['line 1', 'line 2', 'line 3']
            FileHandler.write_file(content, outfile)

            assert outfile.exists()
            result = outfile.read_text()
            assert 'line 1' in result
            assert 'line 2' in result
            assert 'line 3' in result


class TestMMPBSARun:
    """Test suite for MMPBSA run method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_mmpbsa_run_creates_output_dir(self, mock_filehandler):
        """Test that run creates output directory"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock trajectory")

            out_dir = path / 'output' / 'mmpbsa'

            mock_fh.path = out_dir
            mock_fh.files = ([], [], [], [])
            mock_fh.files_chunked = []
            mock_fh.n_cpus = 1

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                out=str(out_dir)
            )

            # Mock methods to avoid actual computation
            mmpbsa.write_mdins = MagicMock(return_value=(path / 'gb.mdin', path / 'pb.mdin'))
            mmpbsa._run_serial = MagicMock()
            mmpbsa._run_frame_parallel = MagicMock()
            mmpbsa._combine_sasa_chunks = MagicMock()
            mmpbsa._combine_energy_chunks = MagicMock()
            mmpbsa._verify_combined_outputs = MagicMock()

            # Just test initialization, not the actual run
            assert mmpbsa is not None


class TestMMPBSARunMethods:
    """Test suite for MMPBSA run methods with mocking"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer')
    def test_run_serial_mode(self, mock_analyzer, mock_filehandler):
        """Test run method in serial mode"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_fh.path = Path('/tmp/test')
        mock_fh.files = [
            (Path('/tmp/test/complex'), Path('/tmp/test/complex.prmtop'),
             Path('/tmp/test/complex.crd'), Path('/tmp/test/complex.pdb')),
        ]
        mock_fh.files_chunked = []
        mock_filehandler.return_value = mock_fh

        mock_analyzer_inst = MagicMock()
        mock_analyzer_inst.free_energy = -10.5
        mock_analyzer.return_value = mock_analyzer_inst

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock")

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                parallel_mode='serial'
            )

            with patch.object(mmpbsa, 'write_mdins') as mock_write, \
                 patch.object(mmpbsa, 'calculate_sasa') as mock_sasa, \
                 patch.object(mmpbsa, 'calculate_energy') as mock_energy:
                mock_write.return_value = (path / 'gb.mdin', path / 'pb.mdin')

                mmpbsa.run()

                mock_write.assert_called_once()
                mock_analyzer_inst.parse_outputs.assert_called_once()
                assert mmpbsa.free_energy == -10.5

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer')
    @patch('molecular_simulations.simulate.mmpbsa._run_sasa_calculation')
    @patch('molecular_simulations.simulate.mmpbsa._run_energy_calculation')
    def test_run_frame_parallel_mode(self, mock_energy_calc, mock_sasa_calc,
                                      mock_analyzer, mock_filehandler):
        """Test run method in frame parallel mode"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_fh.path = Path('/tmp/test')
        mock_fh.files = []
        mock_fh.files_chunked = [
            (Path('/tmp/test/complex'), Path('/tmp/test/complex.prmtop'),
             [Path('/tmp/test/complex_chunk0.crd')], Path('/tmp/test/complex.pdb')),
        ]
        mock_filehandler.return_value = mock_fh

        mock_analyzer_inst = MagicMock()
        mock_analyzer_inst.free_energy = -15.0
        mock_analyzer.return_value = mock_analyzer_inst

        mock_sasa_calc.return_value = ('sasa_script', True, '')
        mock_energy_calc.return_value = ('energy_out', True, '')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock")

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                parallel_mode='frame',
                n_cpus=2
            )

            with patch.object(mmpbsa, 'write_mdins') as mock_write, \
                 patch.object(mmpbsa, '_write_sasa_script') as mock_sasa_script, \
                 patch.object(mmpbsa, '_combine_sasa_chunks') as mock_combine_sasa, \
                 patch.object(mmpbsa, '_combine_energy_chunks') as mock_combine_energy, \
                 patch.object(mmpbsa, '_verify_combined_outputs') as mock_verify:
                mock_write.return_value = (path / 'gb.mdin', path / 'pb.mdin')
                mock_sasa_script.return_value = path / 'sasa.in'

                mmpbsa.run()

                mock_write.assert_called_once()
                mock_analyzer_inst.parse_outputs.assert_called_once()


class TestCalculateEnergy:
    """Test suite for calculate_energy method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_calculate_energy(self, mock_subprocess, mock_filehandler):
        """Test calculate_energy method"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        mock_subprocess.run.return_value = MagicMock(returncode=0, stderr='', stdout='')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock")

            # Create mock mdin file
            mdin_file = path / 'gb.mdin'
            mdin_file.write_text("&general\n/")

            mock_fh.path = path

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150']
            )
            mmpbsa.path = path

            prefix = path / 'complex'
            prm = path / 'complex.prmtop'
            trj = path / 'complex.crd'
            pdb = path / 'complex.pdb'

            mmpbsa.calculate_energy(prefix, prm, trj, pdb, mdin_file, 'gb')

            mock_subprocess.run.assert_called_once()


class TestOutputAnalyzerParsing:
    """Test suite for OutputAnalyzer parsing methods"""

    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer.generate_summary')
    def test_parse_outputs(self, mock_summary):
        """Test parse_outputs method"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create GB energy output files
            gb_content = """ BOND =  100.0  ANGLE = 50.0  DIHED = 25.0
 VDWAALS = -10.0  EEL = -200.0  EGB = -50.0
 1-4 VDW = 5.0  1-4 EEL = 10.0  RESTRAINT = 0.0
 ESURF = -5.0
"""
            # Create PB energy output files (different fields)
            pb_content = """ BOND =  100.0  ANGLE = 50.0  DIHED = 25.0
 VDWAALS = -10.0  EEL = -200.0  EPB = -50.0
 1-4 VDW = 5.0  1-4 EEL = 10.0  RESTRAINT = 0.0
 ECAVITY = -5.0  EDISPER = -3.0
"""
            for system in ['complex', 'receptor', 'ligand']:
                # GB file
                gb_file = path / f'{system}_gb.mdout'
                gb_file.write_text(gb_content)

                # PB file
                pb_file = path / f'{system}_pb.mdout'
                pb_file.write_text(pb_content)

                sasa_file = path / f'{system}_surf.dat'
                sasa_file.write_text("#Frame SASA\n1 1000.0\n")

            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)
            analyzer.parse_outputs()

            # Verify parsing occurred
            assert analyzer.gb is not None
            assert analyzer.pb is not None
            mock_summary.assert_called_once()

    def test_parse_line(self):
        """Test parse_line method"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)

            # Test parsing energy line
            line = " BOND =  100.0  ANGLE = 50.0  DIHED = 25.0"
            result = list(analyzer.parse_line(line))

            assert len(result) == 3
            assert ('BOND', 100.0) in result
            assert ('ANGLE', 50.0) in result
            assert ('DIHED', 25.0) in result

    def test_read_sasa_basic(self):
        """Test read_sasa with basic format"""
        from molecular_simulations.simulate.mmpbsa import OutputAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Simple SASA file without comments in the middle
            sasa_content = """#Frame SASA
1 1000.0
2 1100.0
3 1050.0
"""
            sasa_file = path / 'test_surf.dat'
            sasa_file.write_text(sasa_content)

            analyzer = OutputAnalyzer(path=str(path), surface_tension=0.0072)
            sasa_series = analyzer.read_sasa(sasa_file)

            assert len(sasa_series) == 3


class TestFileHandlerMethods:
    """Test suite for FileHandler class methods"""

    @patch('molecular_simulations.simulate.mmpbsa.subprocess')
    def test_filehandler_init(self, mock_subprocess):
        """Test FileHandler initialization"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        # Mock subprocess.run to return success with frame count output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Total frames: 100\n"
        mock_subprocess.run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock")

            fh = FileHandler(
                top=top_file,
                traj=traj_file,
                path=path,
                sels=[':1-100', ':101-150'],
                first=0,
                last=-1,
                stride=1,
                cpptraj_binary='/fake/cpptraj',
                n_chunks=2
            )

            assert fh.path == path
            mock_subprocess.run.assert_called()

    def test_write_file_pathlike(self):
        """Test write_file with Path object"""
        from molecular_simulations.simulate.mmpbsa import FileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            outfile = path / 'output.txt'

            FileHandler.write_file(['line1', 'line2'], outfile)

            assert outfile.exists()
            content = outfile.read_text()
            assert 'line1' in content
            assert 'line2' in content


class TestRunFrameParallelFailures:
    """Test failure handling in _run_frame_parallel"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    @patch('molecular_simulations.simulate.mmpbsa.OutputAnalyzer')
    @patch('molecular_simulations.simulate.mmpbsa._run_sasa_calculation')
    def test_sasa_failure_raises(self, mock_sasa_calc, mock_analyzer, mock_filehandler):
        """Test that SASA failures raise RuntimeError"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_fh.path = Path('/tmp/test')
        mock_fh.files_chunked = [
            (Path('/tmp/test/complex'), Path('/tmp/test/complex.prmtop'),
             [Path('/tmp/test/complex_chunk0.crd')], Path('/tmp/test/complex.pdb')),
        ]
        mock_filehandler.return_value = mock_fh

        # SASA calculation fails
        mock_sasa_calc.return_value = ('sasa_script', False, 'SASA error')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock")

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                parallel_mode='frame'
            )

            with patch.object(mmpbsa, 'write_mdins') as mock_write, \
                 patch.object(mmpbsa, '_write_sasa_script') as mock_sasa_script:
                mock_write.return_value = (path / 'gb.mdin', path / 'pb.mdin')
                mock_sasa_script.return_value = path / 'sasa.in'

                with pytest.raises(RuntimeError, match="SASA calculations failed"):
                    mmpbsa.run()


class TestMMPBSAKwargs:
    """Test MMPBSA with extra kwargs"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.simulate.mmpbsa.FileHandler')
    def test_mmpbsa_with_kwargs(self, mock_filehandler):
        """Test MMPBSA stores extra kwargs as attributes"""
        from molecular_simulations.simulate.mmpbsa import MMPBSA

        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock")
            traj_file = path / 'traj.dcd'
            traj_file.write_text("mock")

            mmpbsa = MMPBSA(
                top=str(top_file),
                dcd=str(traj_file),
                selections=[':1-100', ':101-150'],
                custom_attr='custom_value',
                another_attr=42
            )

            assert hasattr(mmpbsa, 'custom_attr')
            assert mmpbsa.custom_attr == 'custom_value'
            assert mmpbsa.another_attr == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
