"""
Unit tests for simulate/cph_simulation.py module

This module tests the constant pH simulation setup and ensemble management
classes used for pH-dependent molecular simulations.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, MagicMock, patch, PropertyMock


# Mark tests that don't require OpenMM as unit tests
pytestmark = pytest.mark.unit


class TestConstantPHEnsembleInit:
    """Test suite for ConstantPHEnsemble class initialization."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_ensemble_init_defaults(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble initialization with default parameters.

        Verifies that the class correctly initializes with default pH range
        and temperature values.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {
                'CYS': [0.0, 10.0],
                'ASP': [0.0, 5.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.paths == [path]
            assert ensemble.ref_energies == ref_energies
            assert ensemble.parsl_config is mock_config
            assert ensemble.log_dir == log_dir
            # Default pH range is 0.5 to 13.5 in 1.0 increments
            assert len(ensemble.pHs) == 14
            assert ensemble.pHs[0] == 0.5
            # Temperature should be 300K by default
            assert ensemble.temperature == 300.0 * kelvin

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_ensemble_init_custom_phs(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble with custom pH values."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}
            custom_phs = [4.0, 5.0, 6.0, 7.0, 8.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            assert ensemble.pHs == custom_phs

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_ensemble_init_custom_temperature(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble with custom temperature.

        The temperature is multiplied by kelvin unit internally.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=310.0,
            )

            assert ensemble.temperature == 310.0 * kelvin

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_ensemble_init_with_variant_sel(self, mock_parsl: MagicMock) -> None:
        """Test ConstantPHEnsemble with custom variant selection string.

        The variant_sel parameter allows selecting specific residues for
        titration rather than all titratable residues.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                variant_sel='resid 10:50',
            )

            assert ensemble.variant_sel == 'resid 10:50'

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_ensemble_init_generates_run_id(self, mock_parsl: MagicMock) -> None:
        """Test that initialization generates a unique run ID based on timestamp."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        import re

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # run_id should match format YYYYMMDD_HHMMSS
            assert re.match(r'\d{8}_\d{6}', ensemble.run_id)


class TestConstantPHEnsembleParslManagement:
    """Test suite for Parsl initialization and shutdown."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_initialize_loads_parsl(self, mock_parsl: MagicMock) -> None:
        """Test that initialize() loads the Parsl configuration."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            mock_dfk = MagicMock()
            mock_parsl.load.return_value = mock_dfk

            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.dfk is None
            ensemble.initialize()

            mock_parsl.load.assert_called_once_with(mock_config)
            assert ensemble.dfk is mock_dfk

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_shutdown_cleans_up_parsl(self, mock_parsl: MagicMock) -> None:
        """Test that shutdown() properly cleans up Parsl resources."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            mock_dfk = MagicMock()
            mock_parsl.load.return_value = mock_dfk

            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.initialize()
            ensemble.shutdown()

            mock_dfk.cleanup.assert_called_once()
            mock_parsl.clear.assert_called()
            assert ensemble.dfk is None

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_shutdown_when_not_initialized(self, mock_parsl: MagicMock) -> None:
        """Test that shutdown() handles case when dfk is None."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Should not raise even when dfk is None
            ensemble.shutdown()
            mock_parsl.clear.assert_called()


class TestConstantPHEnsembleParams:
    """Test suite for params property."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_contains_required_keys(self, mock_parsl: MagicMock) -> None:
        """Test params property returns dictionary with all required keys.

        The params dictionary should contain simulation parameters for both
        explicit and implicit solvent, integrators, and pH values.
        """
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Set path attribute as the property expects it
            ensemble.path = path

            params = ensemble.params

            # Check required keys
            assert 'prmtop_file' in params
            assert 'inpcrd_file' in params
            assert 'pH' in params
            assert 'relaxationSteps' in params
            assert 'explicitArgs' in params
            assert 'implicitArgs' in params
            assert 'integrator' in params
            assert 'relaxationIntegrator' in params

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_explicit_args(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct explicit solvent arguments."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.app import PME, HBonds
        from openmm.unit import nanometers, amu

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.path = path
            params = ensemble.params

            expl_args = params['explicitArgs']
            assert expl_args['nonbondedMethod'] == PME
            assert expl_args['nonbondedCutoff'] == 0.9 * nanometers
            assert expl_args['constraints'] == HBonds
            assert expl_args['hydrogenMass'] == 1.5 * amu

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_implicit_args(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct implicit solvent arguments."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.app import CutoffNonPeriodic, HBonds
        from openmm.unit import nanometers

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.path = path
            params = ensemble.params

            impl_args = params['implicitArgs']
            assert impl_args['nonbondedMethod'] == CutoffNonPeriodic
            assert impl_args['nonbondedCutoff'] == 2.0 * nanometers
            assert impl_args['constraints'] == HBonds

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_ph_values(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct pH values from initialization."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}
            custom_phs = [5.0, 6.0, 7.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            ensemble.path = path
            params = ensemble.params

            assert params['pH'] == custom_phs

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_integrator_temperature(self, mock_parsl: MagicMock) -> None:
        """Test params integrator uses correct temperature."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=310.0,
            )

            ensemble.path = path
            params = ensemble.params

            # Integrator should use the ensemble temperature
            integrator = params['integrator']
            # Temperature is the first argument to LangevinIntegrator
            assert integrator.getTemperature() == 310.0 * kelvin

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_relaxation_steps(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct relaxation steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.path = path
            params = ensemble.params

            assert params['relaxationSteps'] == 1000

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_params_file_paths(self, mock_parsl: MagicMock) -> None:
        """Test params contains correct file paths."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.path = path
            params = ensemble.params

            assert params['prmtop_file'] == path / 'system.prmtop'
            assert params['inpcrd_file'] == path / 'system.inpcrd'


class TestConstantPHEnsembleTemperatureHandling:
    """Test suite for temperature unit handling."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_temperature_stored_with_units(self, mock_parsl: MagicMock) -> None:
        """Test that temperature is stored with kelvin units."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=300.0,
            )

            # Temperature should have kelvin units
            assert ensemble.temperature == 300.0 * kelvin
            # Can convert to numeric value
            assert ensemble.temperature.value_in_unit(kelvin) == 300.0

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @pytest.mark.parametrize("temp", [273.15, 300.0, 310.0, 350.0])
    def test_temperature_various_values(
        self, mock_parsl: MagicMock, temp: float
    ) -> None:
        """Test temperature handling with various physiological temperatures."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=temp,
            )

            assert ensemble.temperature.value_in_unit(kelvin) == temp


class TestConstantPHEnsembleMultiplePaths:
    """Test suite for handling multiple simulation paths."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_multiple_paths_stored(self, mock_parsl: MagicMock) -> None:
        """Test that multiple paths are stored correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f'system{i}' for i in range(5)]
            for p in paths:
                p.mkdir()

            log_dir = base / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert len(ensemble.paths) == 5
            assert ensemble.paths == paths

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_single_path(self, mock_parsl: MagicMock) -> None:
        """Test with single path."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert len(ensemble.paths) == 1
            assert ensemble.paths[0] == path


class TestConstantPHEnsembleReferenceEnergies:
    """Test suite for reference energy handling."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_reference_energies_stored(self, mock_parsl: MagicMock) -> None:
        """Test that reference energies are stored correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {
                'CYS': [0.0, 10.0],
                'ASP': [0.0, 5.0],
                'GLU': [0.0, 6.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.ref_energies == ref_energies
            assert 'CYS' in ensemble.ref_energies
            assert 'ASP' in ensemble.ref_energies
            assert 'GLU' in ensemble.ref_energies


class TestConstantPHEnsemblePHRange:
    """Test suite for pH range handling."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_default_ph_range(self, mock_parsl: MagicMock) -> None:
        """Test default pH range is 0.5 to 13.5 in 1.0 steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Default should be [0.5, 1.5, 2.5, ..., 13.5]
            assert len(ensemble.pHs) == 14
            assert ensemble.pHs[0] == 0.5
            assert ensemble.pHs[-1] == 13.5

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_custom_ph_range(self, mock_parsl: MagicMock) -> None:
        """Test custom pH range."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}
            custom_phs = [2.0, 4.0, 6.0, 8.0, 10.0]

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=custom_phs,
            )

            assert ensemble.pHs == custom_phs
            assert len(ensemble.pHs) == 5

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_single_ph(self, mock_parsl: MagicMock) -> None:
        """Test with single pH value."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                pHs=[7.4],  # Single physiological pH
            )

            assert ensemble.pHs == [7.4]
            assert len(ensemble.pHs) == 1


class TestConstantPHEnsembleIntegrators:
    """Test suite for integrator configuration."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_integrator_created(self, mock_parsl: MagicMock) -> None:
        """Test that main integrator is created correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm import LangevinIntegrator

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.path = path
            params = ensemble.params

            assert params['integrator'] is not None
            assert isinstance(params['integrator'], LangevinIntegrator)

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_relaxation_integrator_created(self, mock_parsl: MagicMock) -> None:
        """Test that relaxation integrator is created correctly."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm import LangevinIntegrator

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            ensemble.path = path
            params = ensemble.params

            assert params['relaxationIntegrator'] is not None
            assert isinstance(params['relaxationIntegrator'], LangevinIntegrator)

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_integrators_use_ensemble_temperature(
        self, mock_parsl: MagicMock
    ) -> None:
        """Test both integrators use the ensemble temperature."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kelvin

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                temperature=320.0,
            )

            ensemble.path = path
            params = ensemble.params

            # Both integrators should use 320K
            assert params['integrator'].getTemperature() == 320.0 * kelvin
            assert params['relaxationIntegrator'].getTemperature() == 320.0 * kelvin


class TestConstantPHEnsembleVariantSel:
    """Test suite for variant selection handling."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_variant_sel_none_by_default(self, mock_parsl: MagicMock) -> None:
        """Test that variant_sel is None by default."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            assert ensemble.variant_sel is None

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_variant_sel_custom_string(self, mock_parsl: MagicMock) -> None:
        """Test custom variant selection string."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                variant_sel='resid 10 to 50',
            )

            assert ensemble.variant_sel == 'resid 10 to 50'


class TestConstantPHEnsembleLoadFiles:
    """Test suite for load_files method."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    def test_load_files_returns_topology_and_positions(
        self, mock_parsl: MagicMock
    ) -> None:
        """Test load_files returns topology and positions from AMBER files."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_topology = MagicMock()
        mock_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'
            (path / 'system.prmtop').write_text("mock")
            (path / 'system.inpcrd').write_text("mock")

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Test load_files by mocking at the instance level
            with patch.object(ensemble, 'load_files') as mock_load_files:
                mock_load_files.return_value = (mock_topology, mock_positions)
                top, pos = ensemble.load_files(path)

                assert top is mock_topology
                assert pos is mock_positions
                mock_load_files.assert_called_once_with(path)


class TestConstantPHEnsembleBuildDicts:
    """Test suite for build_dicts method."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @patch('molecular_simulations.simulate.cph_simulation.mda')
    def test_build_dicts_identifies_titratable_residues(
        self, mock_mda: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test build_dicts correctly identifies titratable residues."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble
        from openmm.unit import kilojoules_per_mole

        # Setup mock topology with titratable residues
        mock_residue1 = MagicMock()
        mock_residue1.name = 'CYS'
        mock_residue1.index = 0

        mock_residue2 = MagicMock()
        mock_residue2.name = 'ASP'
        mock_residue2.index = 1

        mock_residue3 = MagicMock()
        mock_residue3.name = 'ALA'  # Non-titratable
        mock_residue3.index = 2

        mock_topology = MagicMock()
        mock_topology.residues.return_value = [mock_residue1, mock_residue2, mock_residue3]

        # Setup mock MDAnalysis universe
        mock_universe = MagicMock()
        mock_protein_sel = MagicMock()
        mock_protein_sel.residues.resids = [0, 1, 2]
        mock_protein_sel.__getitem__ = MagicMock(side_effect=[
            MagicMock(resid=0),  # first residue (terminus)
            MagicMock(resid=2),  # last residue (terminus)
        ])
        mock_universe.select_atoms.return_value = mock_protein_sel
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {
                'CYS': [0.0, 10.0],
                'ASP': [0.0, 5.0],
                'GLU': [0.0, 6.0],
                'LYS': [0.0, 8.0],
                'HIS': [0.0, 4.0, 3.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            variants, reference_energies = ensemble.build_dicts(path, mock_topology)

            # CYS at index 0 is terminus and should be excluded
            # ASP at index 1 should be included
            # ALA at index 2 is terminus and non-titratable
            assert 1 in variants
            assert variants[1] == ['ASH', 'ASP']

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @patch('molecular_simulations.simulate.cph_simulation.mda')
    def test_build_dicts_with_variant_sel(
        self, mock_mda: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test build_dicts uses variant_sel to filter residues."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        # Setup mock topology
        mock_residue1 = MagicMock()
        mock_residue1.name = 'ASP'
        mock_residue1.index = 5

        mock_residue2 = MagicMock()
        mock_residue2.name = 'GLU'
        mock_residue2.index = 10

        mock_topology = MagicMock()
        mock_topology.residues.return_value = [mock_residue1, mock_residue2]

        # Setup mock MDAnalysis
        mock_universe = MagicMock()
        mock_protein_sel = MagicMock()
        mock_protein_sel.residues.resids = [1, 5, 10, 20]
        mock_protein_sel.__getitem__ = MagicMock(side_effect=[
            MagicMock(resid=1),
            MagicMock(resid=20),
        ])

        # Mock variant selection that only includes resid 5
        mock_var_sel = MagicMock()
        mock_var_sel.residues.resids = [5]

        def select_side_effect(selection):
            if 'resid 5' in selection:
                return mock_var_sel
            return mock_protein_sel

        mock_protein_sel.select_atoms.side_effect = select_side_effect
        mock_universe.select_atoms.return_value = mock_protein_sel
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {
                'ASP': [0.0, 5.0],
                'GLU': [0.0, 6.0],
            }

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
                variant_sel='resid 5',
            )

            variants, reference_energies = ensemble.build_dicts(path, mock_topology)

            # Only residue at index 5 should be included
            assert 5 in variants
            # GLU at index 10 should be excluded by variant_sel
            assert 10 not in variants


class TestConstantPHEnsembleRun:
    """Test suite for run method."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @patch('molecular_simulations.simulate.cph_simulation.run_cph_sim')
    def test_run_submits_futures_for_all_paths(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run method submits jobs for all paths."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f'system{i}' for i in range(3)]
            for p in paths:
                p.mkdir()

            log_dir = base / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            # Mock load_files and build_dicts
            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(ensemble, 'load_files', return_value=(mock_topology, mock_positions)):
                with patch.object(ensemble, 'build_dicts', return_value=({}, {})):
                    with patch.object(type(ensemble), 'params', new_callable=PropertyMock) as mock_params:
                        mock_params.return_value = {
                            'prmtop_file': base / 'system.prmtop',
                            'inpcrd_file': base / 'system.inpcrd',
                            'pH': [7.0],
                            'relaxationSteps': 1000,
                            'explicitArgs': {},
                            'implicitArgs': {},
                            'integrator': MagicMock(),
                            'relaxationIntegrator': MagicMock(),
                        }

                        ensemble.run(n_cycles=10, n_steps=100)

            # Should call run_cph_sim for each path
            assert mock_run_cph_sim.call_count == 3
            # Should wait for all futures
            assert mock_future.result.call_count == 3

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @patch('molecular_simulations.simulate.cph_simulation.run_cph_sim')
    def test_run_passes_correct_parameters(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run method passes correct n_cycles and n_steps."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(ensemble, 'load_files', return_value=(mock_topology, mock_positions)):
                with patch.object(ensemble, 'build_dicts', return_value=({}, {})):
                    with patch.object(type(ensemble), 'params', new_callable=PropertyMock) as mock_params:
                        mock_params.return_value = {
                            'prmtop_file': path / 'system.prmtop',
                            'inpcrd_file': path / 'system.inpcrd',
                            'pH': [7.0],
                            'relaxationSteps': 1000,
                            'explicitArgs': {},
                            'implicitArgs': {},
                            'integrator': MagicMock(),
                            'relaxationIntegrator': MagicMock(),
                        }

                        ensemble.run(n_cycles=250, n_steps=750)

            call_args = mock_run_cph_sim.call_args
            assert call_args[0][2] == 250  # n_cycles
            assert call_args[0][3] == 750  # n_steps


class TestConstantPHEnsembleRunWithDefaults:
    """Test suite for run method with default parameters."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @patch('molecular_simulations.simulate.cph_simulation.run_cph_sim')
    def test_run_uses_default_parameters(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run method uses default n_cycles=500 and n_steps=500."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            log_dir = path / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=[path],
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(ensemble, 'load_files', return_value=(mock_topology, mock_positions)):
                with patch.object(ensemble, 'build_dicts', return_value=({}, {})):
                    with patch.object(type(ensemble), 'params', new_callable=PropertyMock) as mock_params:
                        mock_params.return_value = {
                            'prmtop_file': path / 'system.prmtop',
                            'inpcrd_file': path / 'system.inpcrd',
                            'pH': [7.0],
                            'relaxationSteps': 1000,
                            'explicitArgs': {},
                            'implicitArgs': {},
                            'integrator': MagicMock(),
                            'relaxationIntegrator': MagicMock(),
                        }

                        # Call without arguments to use defaults
                        ensemble.run()

            call_args = mock_run_cph_sim.call_args
            assert call_args[0][2] == 500  # default n_cycles
            assert call_args[0][3] == 500  # default n_steps


class TestConstantPHEnsembleLogParams:
    """Test suite for log parameter generation."""

    @patch('molecular_simulations.simulate.cph_simulation.parsl')
    @patch('molecular_simulations.simulate.cph_simulation.run_cph_sim')
    def test_run_generates_unique_task_ids(
        self, mock_run_cph_sim: MagicMock, mock_parsl: MagicMock
    ) -> None:
        """Test run generates unique task IDs for each path."""
        from molecular_simulations.simulate.cph_simulation import ConstantPHEnsemble

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_run_cph_sim.return_value = mock_future

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = [base / f'system{i}' for i in range(3)]
            for p in paths:
                p.mkdir()

            log_dir = base / 'logs'

            mock_config = MagicMock()
            ref_energies = {'CYS': [0.0, 10.0]}

            ensemble = ConstantPHEnsemble(
                paths=paths,
                reference_energies=ref_energies,
                parsl_config=mock_config,
                log_dir=log_dir,
            )

            mock_topology = MagicMock()
            mock_positions = MagicMock()

            with patch.object(ensemble, 'load_files', return_value=(mock_topology, mock_positions)):
                with patch.object(ensemble, 'build_dicts', return_value=({}, {})):
                    with patch.object(type(ensemble), 'params', new_callable=PropertyMock) as mock_params:
                        mock_params.return_value = {
                            'prmtop_file': base / 'system.prmtop',
                            'inpcrd_file': base / 'system.inpcrd',
                            'pH': [7.0],
                            'relaxationSteps': 1000,
                            'explicitArgs': {},
                            'implicitArgs': {},
                            'integrator': MagicMock(),
                            'relaxationIntegrator': MagicMock(),
                        }

                        ensemble.run(n_cycles=10, n_steps=100)

            # Check each call had unique task_id
            task_ids = []
            for call in mock_run_cph_sim.call_args_list:
                log_params = call[0][4]
                task_ids.append(log_params['task_id'])

            assert len(set(task_ids)) == 3
            assert '00000' in task_ids
            assert '00001' in task_ids
            assert '00002' in task_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
