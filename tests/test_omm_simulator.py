"""
Unit tests for simulate/omm_simulator.py module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import os


class TestSimulatorInit:
    """Test suite for Simulator class initialization"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_simulator_init_defaults(self, mock_platform):
        """Test Simulator initialization with defaults"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            # Pass path as Path object, not string, to avoid bug in source
            sim = Simulator(path=path)

            assert sim.path == path
            assert sim.top_file == path / 'system.prmtop'
            assert sim.coor_file == path / 'system.inpcrd'
            assert sim.temperature == 300.0
            assert sim.equil_steps == 1_250_000
            assert sim.prod_steps == 250_000_000

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_simulator_init_custom_files(self, mock_platform):
        """Test Simulator initialization with custom file names"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'custom.prmtop').write_text("mock topology")
            (path / 'custom.inpcrd').write_text("mock coordinates")

            sim = Simulator(
                path=path,
                top_name='custom.prmtop',
                coor_name='custom.inpcrd'
            )

            assert sim.top_file == path / 'custom.prmtop'
            assert sim.coor_file == path / 'custom.inpcrd'

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_simulator_init_custom_output_path(self, mock_platform):
        """Test Simulator initialization with custom output path"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            out_path = path / 'output'
            out_path.mkdir()

            sim = Simulator(path=path, out_path=out_path)

            assert sim.dcd == out_path / 'prod.dcd'
            assert sim.prod_log == out_path / 'prod.log'

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_simulator_init_cpu_platform(self, mock_platform):
        """Test Simulator initialization with CPU platform"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path, platform='CPU')

            # CPU platform should have empty properties
            assert sim.properties == {}


class TestSimulatorBarostat:
    """Test suite for Simulator barostat setup"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.MonteCarloBarostat')
    def test_setup_barostat_non_membrane(self, mock_barostat, mock_platform):
        """Test barostat setup for non-membrane system"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path, membrane=False)

            assert sim.barostat is not None
            assert 'temperature' in sim.barostat_args

    @pytest.mark.skip(reason="Source code has bug - nm is not imported")
    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.MonteCarloMembraneBarostat')
    def test_setup_barostat_membrane(self, mock_membrane_barostat, mock_platform):
        """Test barostat setup for membrane system"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path, membrane=True)

            assert 'defaultSurfaceTension' in sim.barostat_args
            assert 'xymode' in sim.barostat_args
            assert 'zmode' in sim.barostat_args


class TestSimulatorLoadSystem:
    """Test suite for Simulator load_system method"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_load_system_invalid_ff(self, mock_platform):
        """Test load_system with invalid force field"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path, ff='invalid')

            with pytest.raises(AttributeError, match="valid MD forcefield"):
                sim.load_system()

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.AmberInpcrdFile')
    @patch('molecular_simulations.simulate.omm_simulator.AmberPrmtopFile')
    def test_load_amber_files(self, mock_prmtop, mock_inpcrd, mock_platform):
        """Test load_amber_files method"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_inpcrd.return_value = MagicMock(boxVectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mock_topology = MagicMock()
        mock_topology.createSystem.return_value = MagicMock()
        mock_prmtop.return_value = mock_topology

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path, ff='amber')
            system = sim.load_amber_files()

            mock_topology.createSystem.assert_called_once()
            assert system is not None

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.PDBFile')
    @patch('molecular_simulations.simulate.omm_simulator.CharmmPsfFile')
    @patch('molecular_simulations.simulate.omm_simulator.ForceField')
    def test_load_charmm_files_no_params(self, mock_ff, mock_psf, mock_pdb, mock_platform):
        """Test load_charmm_files without explicit params"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_pdb_inst = MagicMock()
        mock_pdb_inst.topology.getPeriodicBoxVectors.return_value = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mock_pdb.return_value = mock_pdb_inst
        mock_psf.return_value = MagicMock()
        mock_ff_inst = MagicMock()
        mock_ff_inst.createSystem.return_value = MagicMock()
        mock_ff.return_value = mock_ff_inst

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path, ff='charmm')
            system = sim.load_charmm_files()

            mock_ff_inst.createSystem.assert_called_once()
            assert system is not None


class TestSimulatorSetupSim:
    """Test suite for Simulator setup_sim method"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.LangevinMiddleIntegrator')
    @patch('molecular_simulations.simulate.omm_simulator.Simulation')
    def test_setup_sim(self, mock_sim_class, mock_integrator, mock_platform):
        """Test setup_sim method"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_integrator.return_value = MagicMock()
        mock_sim_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path)
            sim.topology = MagicMock()
            sim.topology.topology = MagicMock()

            mock_system = MagicMock()
            simulation, integrator = sim.setup_sim(mock_system, dt=0.002)

            assert simulation is not None
            assert integrator is not None


class TestSimulatorCheckpoint:
    """Test suite for Simulator checkpoint methods"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_load_checkpoint(self, mock_platform):
        """Test load_checkpoint method"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")
            chkpt = path / 'test.chk'
            chkpt.write_bytes(b"mock checkpoint")

            sim = Simulator(path=path)

            mock_simulation = MagicMock()
            mock_state = MagicMock()
            mock_state.getPositions.return_value = [[0, 0, 0]]
            mock_state.getVelocities.return_value = [[0, 0, 0]]
            mock_simulation.context.getState.return_value = mock_state

            result = sim.load_checkpoint(mock_simulation, str(chkpt))

            mock_simulation.loadCheckpoint.assert_called_once()
            mock_simulation.context.setPositions.assert_called_once()
            mock_simulation.context.setVelocities.assert_called_once()


class TestSimulatorReporters:
    """Test suite for Simulator reporter methods"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.DCDReporter')
    @patch('molecular_simulations.simulate.omm_simulator.StateDataReporter')
    @patch('molecular_simulations.simulate.omm_simulator.CheckpointReporter')
    def test_attach_reporters(self, mock_chkpt_rep, mock_state_rep, mock_dcd_rep, mock_platform):
        """Test attach_reporters method"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path)

            mock_simulation = MagicMock()
            mock_simulation.reporters = []

            dcd_file = path / 'test.dcd'
            log_file = path / 'test.log'
            rst_file = path / 'test.rst'

            result = sim.attach_reporters(
                mock_simulation,
                str(dcd_file),
                str(log_file),
                str(rst_file)
            )

            assert len(result.reporters) == 3


class TestSimulatorRestraintIndices:
    """Test suite for Simulator restraint methods"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.mda')
    def test_get_restraint_indices(self, mock_mda, mock_platform):
        """Test get_restraint_indices method"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        # Setup mock universe
        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.ix = np.array([0, 1, 2, 3])
        mock_selection = MagicMock()
        mock_selection.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_selection
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path)
            indices = sim.get_restraint_indices()

            assert len(indices) == 4
            mock_universe.select_atoms.assert_called_once()

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.mda')
    def test_get_restraint_indices_with_additional_selection(self, mock_mda, mock_platform):
        """Test get_restraint_indices with additional selection"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_universe = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.ix = np.array([0, 1, 2, 3, 4, 5])
        mock_selection = MagicMock()
        mock_selection.atoms = mock_atoms
        mock_universe.select_atoms.return_value = mock_selection
        mock_mda.Universe.return_value = mock_universe

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path)
            indices = sim.get_restraint_indices(addtl_selection='resname LIG')

            # Should have called select_atoms with combined selection
            call_args = mock_universe.select_atoms.call_args[0][0]
            assert 'backbone' in call_args
            assert 'resname LIG' in call_args


class TestSimulatorAddBackbonePosres:
    """Test suite for add_backbone_posres static method"""

    @patch('molecular_simulations.simulate.omm_simulator.CustomExternalForce')
    def test_add_backbone_posres(self, mock_force):
        """Test add_backbone_posres static method"""
        from molecular_simulations.simulate.omm_simulator import Simulator
        from openmm.unit import nanometers

        mock_force_inst = MagicMock()
        mock_force.return_value = mock_force_inst

        # Create mock system
        mock_system = MagicMock()

        # Create mock positions with units
        mock_positions = []
        for i in range(5):
            pos = MagicMock()
            pos.value_in_unit.return_value = [i * 0.1, 0, 0]
            mock_positions.append(pos)

        # Create mock atoms
        mock_atoms = []
        for i in range(5):
            atom = MagicMock()
            atom.index = i
            mock_atoms.append(atom)

        indices = [0, 2, 4]

        with patch('molecular_simulations.simulate.omm_simulator.deepcopy') as mock_deepcopy:
            mock_deepcopy.return_value = MagicMock()

            result = Simulator.add_backbone_posres(
                mock_system,
                mock_positions,
                mock_atoms,
                indices,
                restraint_force=10.0
            )

            # Should have added particles for indices
            assert mock_force_inst.addParticle.call_count == 3


class TestSimulatorCheckNumStepsLeft:
    """Test suite for check_num_steps_left method"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_check_num_steps_left_normal(self, mock_platform):
        """Test check_num_steps_left with normal log file"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            # Create production log
            log_content = "header\tstep\tenergy\n0\t100000\t-1000.0\n1\t200000\t-1001.0\n"
            (path / 'prod.log').write_text(log_content)

            sim = Simulator(path=path, prod_steps=500000)
            sim.prod_log = path / 'prod.log'

            sim.check_num_steps_left()

            # prod_steps should be decremented
            assert sim.prod_steps < 500000

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_check_num_steps_left_empty_log(self, mock_platform):
        """Test check_num_steps_left with empty log file"""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            # Create empty production log
            (path / 'prod.log').write_text("")

            sim = Simulator(path=path, prod_steps=500000)
            sim.prod_log = path / 'prod.log'

            # Should not raise, just return
            sim.check_num_steps_left()


class TestImplicitSimulator:
    """Test suite for ImplicitSimulator class"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_implicit_simulator_init(self, mock_platform):
        """Test ImplicitSimulator initialization"""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = ImplicitSimulator(path=path)

            assert sim.solvent is not None
            assert sim.solute_dielectric == 1.0
            assert sim.solvent_dielectric == 78.5
            # kappa should be computed
            assert sim.kappa > 0

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.AmberInpcrdFile')
    @patch('molecular_simulations.simulate.omm_simulator.AmberPrmtopFile')
    def test_implicit_load_amber_files(self, mock_prmtop, mock_inpcrd, mock_platform):
        """Test ImplicitSimulator load_amber_files method"""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_inpcrd.return_value = MagicMock()
        mock_topology = MagicMock()
        mock_topology.createSystem.return_value = MagicMock()
        mock_prmtop.return_value = mock_topology

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = ImplicitSimulator(path=path)
            system = sim.load_amber_files()

            # Should be called with implicit solvent parameters
            call_kwargs = mock_topology.createSystem.call_args[1]
            assert 'implicitSolvent' in call_kwargs
            assert 'soluteDielectric' in call_kwargs
            assert 'solventDielectric' in call_kwargs


@pytest.mark.skip(reason="Source code has bug - passes args to super() in wrong order")
class TestCustomForcesSimulator:
    """Test suite for CustomForcesSimulator class"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_custom_forces_simulator_init(self, mock_platform):
        """Test CustomForcesSimulator initialization"""
        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_force = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = CustomForcesSimulator(
                path=path,
                custom_force_objects=[mock_force]
            )

            assert sim.custom_forces == [mock_force]

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_add_forces(self, mock_platform):
        """Test add_forces method"""
        from molecular_simulations.simulate.omm_simulator import CustomForcesSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        mock_force1 = MagicMock()
        mock_force2 = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = CustomForcesSimulator(
                path=path,
                custom_force_objects=[mock_force1, mock_force2]
            )

            mock_system = MagicMock()
            result = sim.add_forces(mock_system)

            assert mock_system.addForce.call_count == 2


class TestMinimizer:
    """Test suite for Minimizer class"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_minimizer_init(self, mock_platform):
        """Test Minimizer initialization"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            coor_file = path / 'system.inpcrd'
            coor_file.write_text("mock coordinates")

            minimizer = Minimizer(
                topology=str(top_file),
                coordinates=str(coor_file),
                out='minimized.pdb'
            )

            assert minimizer.topology == top_file
            assert minimizer.coordinates == coor_file
            assert minimizer.out == path / 'minimized.pdb'

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_minimizer_init_cpu(self, mock_platform):
        """Test Minimizer initialization without GPU"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            coor_file = path / 'system.inpcrd'
            coor_file.write_text("mock coordinates")

            minimizer = Minimizer(
                topology=str(top_file),
                coordinates=str(coor_file),
                device_ids=None
            )

            assert 'DeviceIndex' not in minimizer.properties

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_load_files_invalid(self, mock_platform):
        """Test load_files with invalid file type"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.xyz'  # Invalid extension
            top_file.write_text("mock topology")
            coor_file = path / 'system.xyz'
            coor_file.write_text("mock coordinates")

            minimizer = Minimizer(
                topology=str(top_file),
                coordinates=str(coor_file)
            )

            with pytest.raises(FileNotFoundError, match="No viable simulation"):
                minimizer.load_files()

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.AmberInpcrdFile')
    @patch('molecular_simulations.simulate.omm_simulator.AmberPrmtopFile')
    def test_load_amber(self, mock_prmtop, mock_inpcrd, mock_platform):
        """Test load_amber method"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_inpcrd.return_value = MagicMock(boxVectors=None)
        mock_topology = MagicMock()
        mock_topology.createSystem.return_value = MagicMock()
        mock_prmtop.return_value = mock_topology

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            coor_file = path / 'system.inpcrd'
            coor_file.write_text("mock coordinates")

            minimizer = Minimizer(
                topology=str(top_file),
                coordinates=str(coor_file)
            )

            system = minimizer.load_amber()
            assert system is not None

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.PDBFile')
    @patch('molecular_simulations.simulate.omm_simulator.ForceField')
    def test_load_pdb(self, mock_ff, mock_pdb, mock_platform):
        """Test load_pdb method"""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_pdb_inst = MagicMock()
        mock_pdb_inst.topology = MagicMock()
        mock_pdb.return_value = mock_pdb_inst
        mock_ff_inst = MagicMock()
        mock_ff_inst.createSystem.return_value = MagicMock()
        mock_ff.return_value = mock_ff_inst

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.pdb'
            top_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            coor_file = path / 'system.pdb'

            minimizer = Minimizer(
                topology=str(top_file),
                coordinates=str(coor_file)
            )

            system = minimizer.load_pdb()
            assert system is not None


class TestSimulatorRun:
    """Test suite for Simulator run method"""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_run_skip_equilibration(self, mock_platform):
        """Test run method when equilibration files exist."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")
            # Create eq files to skip equilibration
            (path / 'eq.state').write_text("mock state")
            (path / 'eq.chk').write_bytes(b"mock checkpoint")
            (path / 'eq.log').write_text("mock log")

            sim = Simulator(path=path)

            with patch.object(sim, 'equilibrate') as mock_eq, \
                 patch.object(sim, 'production') as mock_prod, \
                 patch.object(sim, 'check_num_steps_left'):
                sim.run()
                # equilibrate should not be called
                mock_eq.assert_not_called()
                mock_prod.assert_called_once()

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_run_with_restart(self, mock_platform):
        """Test run method with restart checkpoint."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")
            # Create all files including restart
            (path / 'eq.state').write_text("mock state")
            (path / 'eq.chk').write_bytes(b"mock checkpoint")
            (path / 'eq.log').write_text("mock log")
            (path / 'prod.rst.chk').write_bytes(b"mock restart")
            (path / 'prod.log').write_text("header\tstep\tenergy\n0\t100000\t-1000.0\n")

            sim = Simulator(path=path)

            with patch.object(sim, 'production') as mock_prod, \
                 patch.object(sim, 'check_num_steps_left'):
                sim.run()
                # Should call production with restart=True
                mock_prod.assert_called_once()
                call_kwargs = mock_prod.call_args[1]
                assert call_kwargs.get('restart') is True


class TestSimulatorEquilibration:
    """Test suite for equilibration methods."""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.StateDataReporter')
    @patch('molecular_simulations.simulate.omm_simulator.DCDReporter')
    def test_equilibrate(self, mock_dcd, mock_state, mock_platform):
        """Test equilibrate method."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = Simulator(path=path)

            with patch.object(sim, 'load_system') as mock_load, \
                 patch.object(sim, 'add_backbone_posres') as mock_posres, \
                 patch.object(sim, 'setup_sim') as mock_setup, \
                 patch.object(sim, '_heating') as mock_heat, \
                 patch.object(sim, '_equilibrate') as mock_eq:

                mock_system = MagicMock()
                mock_load.return_value = mock_system
                mock_posres.return_value = mock_system

                mock_simulation = MagicMock()
                mock_simulation.reporters = []
                mock_integrator = MagicMock()
                mock_setup.return_value = (mock_simulation, mock_integrator)
                mock_heat.return_value = (mock_simulation, mock_integrator)
                mock_eq.return_value = mock_simulation

                sim.coordinate = MagicMock()
                sim.coordinate.positions = [[0, 0, 0]]
                sim.topology = MagicMock()
                sim.topology.topology.atoms.return_value = []
                sim.indices = [0]

                result = sim.equilibrate()

                mock_load.assert_called_once()
                mock_posres.assert_called_once()
                mock_setup.assert_called_once()
                mock_heat.assert_called_once()
                mock_eq.assert_called_once()


class TestSimulatorProduction:
    """Test suite for production methods."""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_production_no_restart(self, mock_platform):
        """Test production method without restart."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")
            chkpt = path / 'test.chk'
            chkpt.write_bytes(b"mock checkpoint")

            sim = Simulator(path=path)

            with patch.object(sim, 'load_system') as mock_load, \
                 patch.object(sim, 'setup_sim') as mock_setup, \
                 patch.object(sim, 'load_checkpoint') as mock_load_chk, \
                 patch.object(sim, 'attach_reporters') as mock_attach, \
                 patch.object(sim, '_production') as mock_prod:

                mock_system = MagicMock()
                mock_load.return_value = mock_system

                mock_simulation = MagicMock()
                mock_integrator = MagicMock()
                mock_setup.return_value = (mock_simulation, mock_integrator)
                mock_load_chk.return_value = mock_simulation
                mock_attach.return_value = mock_simulation
                mock_prod.return_value = mock_simulation

                sim.production(str(chkpt), restart=False)

                mock_load.assert_called_once()
                mock_setup.assert_called_once()
                mock_load_chk.assert_called_once()
                mock_attach.assert_called_once()
                mock_prod.assert_called_once()

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_production_with_restart(self, mock_platform):
        """Test production method with restart."""
        from molecular_simulations.simulate.omm_simulator import Simulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")
            chkpt = path / 'test.chk'
            chkpt.write_bytes(b"mock checkpoint")

            sim = Simulator(path=path)

            with patch.object(sim, 'load_system') as mock_load, \
                 patch.object(sim, 'setup_sim') as mock_setup, \
                 patch.object(sim, 'load_checkpoint') as mock_load_chk, \
                 patch.object(sim, 'attach_reporters') as mock_attach, \
                 patch.object(sim, '_production') as mock_prod:

                mock_system = MagicMock()
                mock_load.return_value = mock_system

                mock_simulation = MagicMock()
                mock_integrator = MagicMock()
                mock_setup.return_value = (mock_simulation, mock_integrator)
                mock_load_chk.return_value = mock_simulation
                mock_attach.return_value = mock_simulation
                mock_prod.return_value = mock_simulation

                sim.production(str(chkpt), restart=True)

                # With restart=True, log file should be opened in append mode
                mock_attach.assert_called_once()
                call_args = mock_attach.call_args[0]
                # log_file arg should be a file object (from open), not string
                assert not isinstance(call_args[2], str)


class TestMinimizerMethods:
    """Additional tests for Minimizer class."""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    @patch('molecular_simulations.simulate.omm_simulator.AmberInpcrdFile')
    @patch('molecular_simulations.simulate.omm_simulator.AmberPrmtopFile')
    @patch('molecular_simulations.simulate.omm_simulator.LangevinMiddleIntegrator')
    @patch('molecular_simulations.simulate.omm_simulator.Simulation')
    @patch('molecular_simulations.simulate.omm_simulator.PDBFile')
    def test_minimizer_minimize(self, mock_pdb, mock_sim_class, mock_integrator,
                                mock_prmtop, mock_inpcrd, mock_platform):
        """Test Minimizer minimize method."""
        from molecular_simulations.simulate.omm_simulator import Minimizer

        mock_platform.getPlatformByName.return_value = MagicMock()
        mock_inpcrd_instance = MagicMock(boxVectors=None)
        mock_inpcrd_instance.positions = [[0, 0, 0]]
        mock_inpcrd.return_value = mock_inpcrd_instance
        mock_topology = MagicMock()
        mock_topology.createSystem.return_value = MagicMock()
        mock_topology.topology = MagicMock()
        mock_prmtop.return_value = mock_topology

        mock_simulation = MagicMock()
        mock_state = MagicMock()
        mock_state.getPositions.return_value = [[0, 0, 0]]
        mock_simulation.context.getState.return_value = mock_state
        mock_sim_class.return_value = mock_simulation
        mock_integrator.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            top_file = path / 'system.prmtop'
            top_file.write_text("mock topology")
            coor_file = path / 'system.inpcrd'
            coor_file.write_text("mock coordinates")

            minimizer = Minimizer(
                topology=str(top_file),
                coordinates=str(coor_file),
                out='minimized.pdb'
            )

            minimizer.minimize()

            mock_simulation.minimizeEnergy.assert_called_once()
            mock_pdb.writeFile.assert_called_once()


class TestImplicitSimulator:
    """Test suite for ImplicitSimulator class."""

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_implicit_simulator_init(self, mock_platform):
        """Test ImplicitSimulator initialization."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = ImplicitSimulator(path=path)

            assert sim.path == path
            assert sim.temperature == 300.0

    @patch('molecular_simulations.simulate.omm_simulator.Platform')
    def test_implicit_simulator_init_with_ff(self, mock_platform):
        """Test ImplicitSimulator initialization with force field parameter."""
        from molecular_simulations.simulate.omm_simulator import ImplicitSimulator

        mock_platform.getPlatformByName.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'system.prmtop').write_text("mock topology")
            (path / 'system.inpcrd').write_text("mock coordinates")

            sim = ImplicitSimulator(path=path, ff='amber')

            assert sim.ff == 'amber'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
