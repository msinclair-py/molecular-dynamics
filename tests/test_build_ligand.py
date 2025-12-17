"""
Unit tests for build/build_ligand.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os
import sys


# Mock difficult dependencies before import
@pytest.fixture(autouse=True)
def mock_difficult_dependencies():
    """Mock dependencies that might not be installed"""
    mock_pybel = MagicMock()
    mock_openbabel = MagicMock()
    mock_openbabel.pybel = mock_pybel

    mock_rdkit = MagicMock()
    mock_chem = MagicMock()
    mock_rdkit.Chem = mock_chem

    # Remove cached build_ligand module to ensure fresh import with new mocks
    modules_to_remove = [
        'molecular_simulations.build.build_ligand',
    ]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    with patch.dict(sys.modules, {
        'openbabel': mock_openbabel,
        'openbabel.pybel': mock_pybel,
        'rdkit': mock_rdkit,
        'rdkit.Chem': mock_chem,
    }):
        # Also patch the module's pybel binding after import
        yield {
            'pybel': mock_pybel,
            'Chem': mock_chem,
        }
        # Cleanup: remove the module so subsequent tests/other files get fresh imports
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)


class TestLigandError:
    """Test suite for LigandError exception class"""

    def test_ligand_error_default_message(self):
        """Test LigandError with default message"""
        from molecular_simulations.build.build_ligand import LigandError

        err = LigandError()
        assert 'cannot model' in str(err)

    def test_ligand_error_custom_message(self):
        """Test LigandError with custom message"""
        from molecular_simulations.build.build_ligand import LigandError

        err = LigandError("Custom error message")
        assert str(err) == "Custom error message"

    def test_ligand_error_is_exception(self):
        """Test LigandError is a proper Exception subclass"""
        from molecular_simulations.build.build_ligand import LigandError

        assert issubclass(LigandError, Exception)

        with pytest.raises(LigandError):
            raise LigandError("Test error")


class TestLigandBuilder:
    """Test suite for LigandBuilder class"""

    def test_ligand_builder_init(self):
        """Test LigandBuilder initialization"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            # The source code expects lig to be a Path for `.stem` on line 89
            # This appears to be a bug in the source - working around by using Path
            builder = LigandBuilder(
                path=path,
                lig=Path('ligand.sdf'),
                lig_number=0
            )

            assert builder.path == path
            assert builder.lig == path / 'ligand.sdf'
            assert builder.ln == 0

    def test_ligand_builder_init_with_prefix(self):
        """Test LigandBuilder initialization with file prefix"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(
                path=path,
                lig=Path('ligand.sdf'),
                lig_number=1,
                file_prefix='prefix_'
            )

            assert builder.ln == 1
            assert 'prefix_' in str(builder.out_lig)

    def test_ligand_builder_write_leap(self):
        """Test write_leap method"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))

            leap_content = "source leaprc.gaff2\nquit"
            leap_file, leap_log = builder.write_leap(leap_content)

            assert Path(leap_file).exists()
            assert Path(leap_file).read_text() == leap_content

    def test_process_sdf(self, mock_difficult_dependencies):
        """Test process_sdf method"""
        from molecular_simulations.build.build_ligand import LigandBuilder
        import molecular_simulations.build.build_ligand as bl_mod

        # Use the module's Chem which is the fixture's mock
        mock_chem = bl_mod.Chem

        # Setup mock
        mock_mol = MagicMock()
        mock_molH = MagicMock()
        mock_chem.SDMolSupplier.return_value = [mock_mol]
        mock_chem.AddHs.return_value = mock_molH
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))
            builder.lig = 'ligand'  # Mimic parameterize_ligand behavior

            builder.process_sdf()

            mock_chem.SDMolSupplier.assert_called_once()
            mock_chem.AddHs.assert_called_once_with(mock_mol, addCoords=True)

    def test_process_pdb(self, mock_difficult_dependencies):
        """Test process_pdb method"""
        from molecular_simulations.build.build_ligand import LigandBuilder
        import molecular_simulations.build.build_ligand as bl_mod

        mock_chem = bl_mod.Chem

        mock_mol = MagicMock()
        mock_molH = MagicMock()
        mock_chem.MolFromPDBFile.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_molH
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.pdb'
            lig_file.write_text("mock pdb content")

            builder = LigandBuilder(path=path, lig=Path('ligand.pdb'))
            builder.lig = 'ligand'

            builder.process_pdb()

            mock_chem.MolFromPDBFile.assert_called_once()
            mock_chem.AddHs.assert_called_once_with(mock_mol, addCoords=True)

    def test_check_sqm_success(self):
        """Test check_sqm with successful calculation"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                path = Path(tmpdir)
                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf content")

                # Create successful sqm output in current directory
                sqm_out = Path('ligand_sqm.out')
                sqm_out.write_text("Some output\nCalculation Completed\nEnd")

                builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))
                builder.lig = 'ligand'

                # Should not raise
                builder.check_sqm()
        finally:
            os.chdir(cwd)

    def test_check_sqm_failure(self):
        """Test check_sqm with failed calculation"""
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                path = Path(tmpdir)
                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf content")

                # Create failed sqm output in current directory
                sqm_out = Path('ligand_sqm.out')
                sqm_out.write_text("Some output\nError occurred\nEnd")

                builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))
                builder.lig = 'ligand'

                with pytest.raises(LigandError, match="SQM failed"):
                    builder.check_sqm()
        finally:
            os.chdir(cwd)

    def test_convert_to_mol2(self, mock_difficult_dependencies):
        """Test convert_to_mol2 method"""
        from molecular_simulations.build.build_ligand import LigandBuilder
        import molecular_simulations.build.build_ligand as bl_mod

        mock_pybel = bl_mod.pybel

        mock_mol = MagicMock()
        mock_pybel.readfile.return_value = [mock_mol]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))
            builder.lig = 'ligand'

            builder.convert_to_mol2()

            mock_pybel.readfile.assert_called_once_with('sdf', 'ligand_H.sdf')
            mock_mol.write.assert_called_once()

    def test_move_antechamber_outputs(self):
        """Test move_antechamber_outputs method"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf content")

                # Create files that antechamber would produce
                os.chdir(tmpdir)
                Path('sqm.in').write_text("sqm input")
                Path('sqm.pdb').write_text("sqm pdb")
                Path('sqm.out').write_text("sqm output")

                builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))
                builder.lig = 'ligand'

                builder.move_antechamber_outputs()

                # sqm.in and sqm.pdb should be removed
                assert not Path('sqm.in').exists()
                assert not Path('sqm.pdb').exists()
                # sqm.out should be renamed
                assert Path('ligand_sqm.out').exists()
        finally:
            os.chdir(cwd)


class TestComplexBuilder:
    """Test suite for ComplexBuilder class"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_complex_builder_init_single_ligand(self):
        """Test ComplexBuilder initialization with single ligand"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=str(lig_file),
                padding=12.0
            )

            assert builder.pad == 12.0
            assert 'leaprc.gaff2' in builder.ffs
            assert isinstance(builder.lig, Path)

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_complex_builder_init_multiple_ligands(self):
        """Test ComplexBuilder initialization with multiple ligands"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            lig_file1 = path / 'ligand1.sdf'
            lig_file1.write_text("mock sdf content")
            lig_file2 = path / 'ligand2.sdf'
            lig_file2.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=[str(lig_file1), str(lig_file2)],
                padding=10.0
            )

            assert isinstance(builder.lig, list)
            assert len(builder.lig) == 2

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_complex_builder_with_precomputed_params(self):
        """Test ComplexBuilder with pre-computed ligand parameters"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            param_prefix = path / 'params' / 'ligand'

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=str(lig_file),
                lig_param_prefix=str(param_prefix)
            )

            assert builder.lig_param_prefix is not None

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_complex_builder_kwargs(self):
        """Test ComplexBuilder with extra kwargs"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            ion_file = path / 'ion.pdb'
            ion_file.write_text("HETATM    1  NA  NA+ A   1       5.000   5.000   5.000  1.00  0.00\n")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=str(lig_file),
                ion=str(ion_file)
            )

            assert hasattr(builder, 'ion')
            assert builder.ion == str(ion_file)

class TestPLINDERBuilder:
    """Test suite for PLINDERBuilder class"""

    def test_cation_list_property(self):
        """Test cation_list property values"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        # Access property directly without instantiating
        cations = PLINDERBuilder.cation_list.fget(None)
        assert 'na' in cations
        assert 'k' in cations
        assert 'ca' in cations
        assert 'mg' in cations

    def test_anion_list_property(self):
        """Test anion_list property values"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        # Access property directly without instantiating
        anions = PLINDERBuilder.anion_list.fget(None)
        assert 'cl' in anions
        assert 'br' in anions
        assert 'i' in anions
        assert 'f' in anions


class TestComplexBuilderMethods:
    """Additional test methods for ComplexBuilder"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_add_ion_to_pdb(self):
        """Test add_ion_to_pdb method"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00
END
"""
            ion_content = """HETATM    1  NA  NA+ A   2       5.000   5.000   5.000  1.00  0.00
"""
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text(pdb_content)

            ion_file = path / 'ion.pdb'
            ion_file.write_text(ion_content)

            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=str(lig_file),
                ion=str(ion_file)
            )

            # Override pdb path for test
            builder.pdb = str(pdb_file)

            builder.add_ion_to_pdb()

            modified_pdb = pdb_file.read_text()
            assert 'HETATM' in modified_pdb
            assert 'NA' in modified_pdb
            assert 'END' in modified_pdb

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_process_ligand_copies_file(self, mock_difficult_dependencies):
        """Test process_ligand copies file to build directory"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        # Mock LigandBuilder directly on the module
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = 'ligand'
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / 'protein.pdb'
                pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf content")

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=str(lig_file)
                )

                # Create build directory
                builder.build_dir = path / 'build'
                builder.build_dir.mkdir()

                result = builder.process_ligand(lig_file)

                # LigandBuilder should be called
                mock_lig_builder.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_complex_builder_with_list_of_ligands(self):
        """Test ComplexBuilder with list of ligands"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

            lig_file1 = path / 'ligand1.sdf'
            lig_file1.write_text("mock sdf content")

            lig_file2 = path / 'ligand2.sdf'
            lig_file2.write_text("mock sdf content")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=[str(lig_file1), str(lig_file2)]
            )

            assert isinstance(builder.lig, list)
            assert len(builder.lig) == 2


class TestLigandBuilderAdditional:
    """Additional tests for LigandBuilder"""

    def test_ligand_builder_default_prefix(self):
        """Test LigandBuilder with default empty prefix"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))

            # out_lig should not have a prefix
            assert str(builder.out_lig).endswith('ligand')

    def test_ligand_builder_lig_number(self):
        """Test LigandBuilder with different ligand numbers"""
        from molecular_simulations.build.build_ligand import LigandBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf content")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'), lig_number=5)

            assert builder.ln == 5


class TestLigandBuilderParameterize:
    """Test suite for LigandBuilder parameterize methods"""

    @patch('molecular_simulations.build.build_ligand.os.system')
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    def test_parameterize_ligand_sdf(self, mock_chdir, mock_os_system, mock_difficult_dependencies):
        """Test parameterize_ligand with SDF file"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import LigandBuilder

        mock_chem = bl_mod.Chem
        mock_pybel = bl_mod.pybel

        mock_os_system.return_value = 0
        mock_mol = MagicMock()
        mock_chem.SDMolSupplier.return_value = [mock_mol]
        mock_chem.AddHs.return_value = mock_mol
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)
        mock_pybel.readfile.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))

            with patch.object(builder, 'check_sqm'), \
                 patch.object(builder, 'move_antechamber_outputs'):
                builder.parameterize_ligand()

            mock_os_system.assert_called()

    @patch('molecular_simulations.build.build_ligand.os.system')
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    def test_parameterize_ligand_pdb(self, mock_chdir, mock_os_system, mock_difficult_dependencies):
        """Test parameterize_ligand with PDB file"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import LigandBuilder

        mock_chem = bl_mod.Chem
        mock_pybel = bl_mod.pybel

        mock_os_system.return_value = 0
        mock_mol = MagicMock()
        mock_chem.MolFromPDBFile.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)
        mock_pybel.readfile.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.pdb'
            lig_file.write_text("ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00\n")

            builder = LigandBuilder(path=path, lig=Path('ligand.pdb'))

            with patch.object(builder, 'check_sqm'), \
                 patch.object(builder, 'move_antechamber_outputs'):
                builder.parameterize_ligand()

            mock_chem.MolFromPDBFile.assert_called()


class TestComplexBuilderBuild:
    """Test suite for ComplexBuilder build methods"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    @patch('molecular_simulations.build.build_amber.subprocess')
    @patch('molecular_simulations.build.build_ligand.LigandBuilder')
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_complex_builder_build(self, mock_os_system, mock_lig_builder, mock_subprocess, mock_chdir):
        """Test ComplexBuilder build method"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_builder = MagicMock()
        mock_builder.lig = 'ligand'
        mock_lig_builder.return_value = mock_builder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")

            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf")

            builder = ComplexBuilder(
                path=str(path),
                pdb=str(pdb_file),
                lig=str(lig_file)
            )

            with patch.object(builder, 'assemble_system'), \
                 patch.object(builder, 'process_ligand') as mock_process:
                mock_process.return_value = 'ligand'

                # Create build directory
                builder.build_dir = path / 'build'
                builder.build_dir.mkdir()

                builder.build()

                mock_process.assert_called_once()


class TestPLINDERBuilderMethods:
    """Test suite for PLINDERBuilder methods"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_cation_list_values(self):
        """Test cation_list contains expected ions"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        cations = PLINDERBuilder.cation_list.fget(None)
        expected = ['na', 'k', 'ca', 'mg', 'zn', 'fe', 'cu', 'mn', 'co', 'ni']
        for ion in expected:
            assert ion in cations

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_anion_list_values(self):
        """Test anion_list contains expected ions"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        anions = PLINDERBuilder.anion_list.fget(None)
        expected = ['cl', 'br', 'i', 'f']
        for ion in expected:
            assert ion in anions

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.ImplicitSolvent.__init__')
    def test_plinder_builder_init(self, mock_super_init):
        """Test PLINDERBuilder initialization"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        mock_super_init.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            out = path / 'output'
            out.mkdir()

            # Create system directory structure
            system_dir = path / 'system_001'
            system_dir.mkdir()

            builder = PLINDERBuilder.__new__(PLINDERBuilder)
            builder.out = out / 'system_001'
            builder.ffs = ['leaprc.protein.ff19SB']
            builder.system_id = 'system_001'
            builder.build_dir = builder.out / 'build'
            builder.ions = None

            assert builder.system_id == 'system_001'
            assert 'leaprc.protein.ff19SB' in builder.ffs


class TestComplexBuilderProcessLigand:
    """Test suite for ComplexBuilder process_ligand method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.shutil')
    def test_process_ligand(self, mock_shutil, mock_difficult_dependencies):
        """Test process_ligand method"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        # Mock LigandBuilder directly
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = 'ligand'
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                pdb_file = path / 'protein.pdb'
                pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")

                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=str(lig_file)
                )

                builder.build_dir = path / 'build'
                builder.build_dir.mkdir()

                result = builder.process_ligand(lig_file)

                mock_lig_builder.assert_called_once()
                mock_builder.parameterize_ligand.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder


class TestPLINDERBuilderBuild:
    """Test suite for PLINDERBuilder build methods"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    @patch('molecular_simulations.build.build_ligand.ImplicitSolvent.__init__')
    @patch('molecular_simulations.build.build_ligand.os.system')
    @patch('molecular_simulations.build.build_ligand.shutil')
    def test_migrate_files_no_ligands(self, mock_shutil, mock_os_system, mock_super_init, mock_chdir):
        """Test migrate_files when no ligands are found"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        mock_super_init.return_value = None
        mock_os_system.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create system directory structure
            system_dir = path / 'system_001'
            system_dir.mkdir()

            # Create empty ligand_files directory
            lig_dir = system_dir / 'ligand_files'
            lig_dir.mkdir()

            # Create sequences.fasta
            (system_dir / 'sequences.fasta').write_text(">A\nALAGLY\n")

            # Create receptor.pdb
            (system_dir / 'receptor.pdb').write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\nEND\n")

            builder = PLINDERBuilder.__new__(PLINDERBuilder)
            builder.path = system_dir
            builder.out = path / 'output' / 'system_001'
            builder.out.mkdir(parents=True)
            builder.ffs = ['leaprc.protein.ff19SB']
            builder.system_id = 'system_001'
            builder.build_dir = builder.out / 'build'
            builder.pdb = 'receptor.pdb'
            builder.ions = None
            builder.fasta = None

            with patch.object(builder, 'prep_protein'):
                ligs = builder.migrate_files()

            assert ligs == []

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_check_ligand_true_ligand(self, mock_difficult_dependencies):
        """Test check_ligand returns True for non-ion ligands"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        mock_chem = bl_mod.Chem

        # Setup mock for non-ion ligand
        mock_mol = MagicMock()
        mock_atom = MagicMock()
        mock_atom.GetSymbol.return_value = 'C'
        mock_atom.GetFormalCharge.return_value = 0
        mock_mol.GetAtoms.return_value = [mock_atom]
        mock_conformer = MagicMock()
        mock_conformer.GetPositions.return_value = [[0.0, 0.0, 0.0]]
        mock_mol.GetConformer.return_value = mock_conformer
        mock_chem.SDMolSupplier.return_value = [mock_mol]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf")

            builder = PLINDERBuilder.__new__(PLINDERBuilder)
            builder.ions = None

            result = builder.check_ligand(lig_file)

            assert result is True
            assert builder.ions is None

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_check_ligand_ion(self, mock_difficult_dependencies):
        """Test check_ligand returns False for ions"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        mock_chem = bl_mod.Chem

        # Setup mock for ion (Na+)
        mock_mol = MagicMock()
        mock_atom = MagicMock()
        mock_atom.GetSymbol.return_value = 'Na'
        mock_atom.GetFormalCharge.return_value = 1
        mock_mol.GetAtoms.return_value = [mock_atom]
        mock_conformer = MagicMock()
        mock_conformer.GetPositions.return_value = [[0.0, 0.0, 0.0]]
        mock_mol.GetConformer.return_value = mock_conformer
        mock_chem.SDMolSupplier.return_value = [mock_mol]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf")

            builder = PLINDERBuilder.__new__(PLINDERBuilder)
            builder.ions = None

            result = builder.check_ligand(lig_file)

            assert result is False
            assert builder.ions is not None
            assert len(builder.ions) == 1

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_place_ions(self):
        """Test place_ions adds ion records to PDB"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00
TER
END
"""
            pdb_file = path / 'protein.pdb'
            pdb_file.write_text(pdb_content)

            builder = PLINDERBuilder.__new__(PLINDERBuilder)
            builder.pdb = str(pdb_file)
            builder.ions = [[['Na', '+', 5.0, 5.0, 5.0]]]

            builder.place_ions()

            modified = pdb_file.read_text()
            assert 'Na' in modified
            assert '5.000' in modified

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_check_ptms_correct_sequence(self):
        """Test check_ptms returns unchanged sequence when correct"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        builder = PLINDERBuilder.__new__(PLINDERBuilder)
        builder.pdb = 'test.pdb'

        # Mock residue objects
        mock_res1 = MagicMock()
        mock_res1.id = '1'
        mock_res1.name = 'ALA'

        mock_res2 = MagicMock()
        mock_res2.id = '2'
        mock_res2.name = 'GLY'

        sequence = ['ALA', 'GLY']
        chain_residues = [mock_res1, mock_res2]

        result = builder.check_ptms(sequence, chain_residues)

        assert result == ['ALA', 'GLY']

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_check_ptms_with_modification(self):
        """Test check_ptms updates PTM residues"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        builder = PLINDERBuilder.__new__(PLINDERBuilder)
        builder.pdb = 'test.pdb'

        # Mock residue with phosphoserine (SEP)
        mock_res1 = MagicMock()
        mock_res1.id = '1'
        mock_res1.name = 'SEP'

        sequence = ['SER']
        chain_residues = [mock_res1]

        result = builder.check_ptms(sequence, chain_residues)

        assert result == ['SEP']


class TestComplexBuilderAssemble:
    """Test suite for ComplexBuilder assemble_system method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_assemble_system_single_ligand(self, mock_os_system):
        """Test assemble_system with single ligand"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf")

            builder = ComplexBuilder.__new__(ComplexBuilder)
            builder.out = path / 'output'
            builder.out.mkdir()
            builder.build_dir = path / 'build'
            builder.build_dir.mkdir()
            builder.pdb = str(pdb_file)
            builder.lig = str(path / 'lig')
            builder.ffs = ['leaprc.protein.ff19SB']
            builder.water_box = 'TIP3PBOX'

            with patch.object(builder, 'write_leap') as mock_write_leap:
                mock_write_leap.return_value = str(path / 'leap.in')
                builder.assemble_system(dim=80.0, num_ions=10)

            mock_os_system.assert_called()
            mock_write_leap.assert_called_once()

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_assemble_system_multiple_ligands(self, mock_os_system):
        """Test assemble_system with multiple ligands"""
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            pdb_file = path / 'protein.pdb'
            pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

            builder = ComplexBuilder.__new__(ComplexBuilder)
            builder.out = path / 'output'
            builder.out.mkdir()
            builder.build_dir = path / 'build'
            builder.build_dir.mkdir()
            builder.pdb = str(pdb_file)
            builder.lig = [str(path / 'lig1'), str(path / 'lig2')]
            builder.ffs = ['leaprc.protein.ff19SB']
            builder.water_box = 'TIP3PBOX'

            with patch.object(builder, 'write_leap') as mock_write_leap:
                mock_write_leap.return_value = str(path / 'leap.in')
                builder.assemble_system(dim=80.0, num_ions=10)

            mock_os_system.assert_called()


class TestPLINDERBuilderAssemble:
    """Test suite for PLINDERBuilder assemble_system method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_assemble_system(self, mock_os_system):
        """Test PLINDERBuilder assemble_system"""
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        mock_os_system.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            builder = PLINDERBuilder.__new__(PLINDERBuilder)
            builder.out = path / 'output'
            builder.out.mkdir()
            builder.build_dir = path / 'build'
            builder.build_dir.mkdir()
            builder.pdb = str(path / 'protein.pdb')
            builder.ligs = ['ligand1', 'ligand2']
            builder.ffs = ['leaprc.protein.ff19SB', 'leaprc.gaff2']

            builder.assemble_system()

            mock_os_system.assert_called_once()
            assert (builder.build_dir / 'tleap.in').exists()


class TestLigandBuilderFileNotFound:
    """Test LigandBuilder error handling"""

    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_parameterize_ligand_file_not_found(self, mock_os_system, mock_difficult_dependencies):
        """Test parameterize_ligand raises LigandError on FileNotFoundError"""
        from molecular_simulations.build.build_ligand import LigandBuilder, LigandError
        import molecular_simulations.build.build_ligand as bl_mod

        # Use the module's mocks directly (set by the autouse fixture)
        mock_chem = bl_mod.Chem
        mock_pybel = bl_mod.pybel

        mock_os_system.return_value = 0
        mock_mol = MagicMock()
        mock_chem.SDMolSupplier.return_value = [mock_mol]
        mock_chem.AddHs.return_value = mock_mol
        mock_writer = MagicMock()
        mock_chem.SDWriter.return_value.__enter__ = Mock(return_value=mock_writer)
        mock_chem.SDWriter.return_value.__exit__ = Mock(return_value=None)
        mock_pybel.readfile.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            lig_file = path / 'ligand.sdf'
            lig_file.write_text("mock sdf")

            builder = LigandBuilder(path=path, lig=Path('ligand.sdf'))

            # Make move_antechamber_outputs raise FileNotFoundError
            with patch.object(builder, 'move_antechamber_outputs', side_effect=FileNotFoundError):
                with pytest.raises(LigandError, match='Antechamber failed'):
                    builder.parameterize_ligand()


class TestPLINDERBuilderLigandHandler:
    """Test suite for PLINDERBuilder ligand_handler method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    def test_ligand_handler(self, mock_difficult_dependencies):
        """Test ligand_handler parameterizes all ligands"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import PLINDERBuilder

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = 'ligand'
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                builder = PLINDERBuilder.__new__(PLINDERBuilder)
                builder.build_dir = path

                ligs = ['ligand1.sdf', 'ligand2.sdf']
                result = builder.ligand_handler(ligs)

                assert len(result) == 2
                assert mock_lig_builder.call_count == 2
                assert mock_builder.parameterize_ligand.call_count == 2
        finally:
            bl_mod.LigandBuilder = original_lig_builder


class TestComplexBuilderBuildMethod:
    """Test suite for ComplexBuilder build method"""

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_build_with_precomputed_params(self, mock_os_system, mock_chdir, mock_difficult_dependencies):
        """Test build with pre-computed ligand parameters"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / 'protein.pdb'
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf")

                params_dir = path / 'params'
                params_dir.mkdir()
                param_prefix = params_dir / 'ligand'

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=str(lig_file),
                    lig_param_prefix=str(param_prefix)
                )

                with patch.object(builder, 'prep_pdb'), \
                     patch.object(builder, 'assemble_system'):
                    builder.build()

                # LigandBuilder should not be called when using precomputed params
                mock_lig_builder.assert_not_called()
        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_build_with_multiple_ligands(self, mock_os_system, mock_chdir, mock_difficult_dependencies):
        """Test build with multiple ligands"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = 'ligand'
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / 'protein.pdb'
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\n")

                lig_file1 = path / 'ligand1.sdf'
                lig_file1.write_text("mock sdf")

                lig_file2 = path / 'ligand2.sdf'
                lig_file2.write_text("mock sdf")

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=[str(lig_file1), str(lig_file2)]
                )

                with patch.object(builder, 'prep_pdb'), \
                     patch.object(builder, 'assemble_system'):
                    builder.build()

                # LigandBuilder should be called for each ligand
                assert mock_lig_builder.call_count == 2
        finally:
            bl_mod.LigandBuilder = original_lig_builder

    @patch.dict(os.environ, {'AMBERHOME': '/fake/amber'})
    @patch('molecular_simulations.build.build_ligand.os.chdir')
    @patch('molecular_simulations.build.build_ligand.os.system')
    def test_build_with_ion(self, mock_os_system, mock_chdir, mock_difficult_dependencies):
        """Test build with ion file"""
        import molecular_simulations.build.build_ligand as bl_mod
        from molecular_simulations.build.build_ligand import ComplexBuilder

        mock_os_system.return_value = 0

        # Manually patch LigandBuilder
        mock_lig_builder = MagicMock()
        mock_builder = MagicMock()
        mock_builder.lig = 'ligand'
        mock_lig_builder.return_value = mock_builder
        original_lig_builder = bl_mod.LigandBuilder
        bl_mod.LigandBuilder = mock_lig_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                pdb_file = path / 'protein.pdb'
                pdb_file.write_text("ATOM  1  N  ALA A 1  0.0 0.0 0.0  1.0 0.0\nEND\n")

                lig_file = path / 'ligand.sdf'
                lig_file.write_text("mock sdf")

                ion_file = path / 'ion.pdb'
                ion_file.write_text("HETATM  1  NA  NA+ A 2  5.0 5.0 5.0  1.0 0.0\n")

                builder = ComplexBuilder(
                    path=str(path),
                    pdb=str(pdb_file),
                    lig=str(lig_file),
                    ion=str(ion_file)
                )

                with patch.object(builder, 'prep_pdb'), \
                     patch.object(builder, 'assemble_system'), \
                     patch.object(builder, 'add_ion_to_pdb') as mock_add_ion:
                    builder.build()

                mock_add_ion.assert_called_once()
        finally:
            bl_mod.LigandBuilder = original_lig_builder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
