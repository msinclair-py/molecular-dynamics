from copy import deepcopy
import MDAnalysis as mda
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.app.internal.singleton import Singleton
import os
from pathlib import Path
from typing import Union

PathLike = Union[Path, str]
OptPath = Union[Path, str, None]

class Simulator:
    """
    Class for performing OpenMM simulations on AMBER FF inputs. Inputs must conform
    to naming conventions found below in the init.

    Arguments:
        path (PathLike): Path to simulation inputs, same as output path.
        equil_steps (int): Defaults to 1,250,000 (2.5 ns). Number of simulation timesteps to
            perform equilibration for (2fs timestep).
        prod_steps (int): Defaults to 250,000,000 (1 µs). Number of simulation timesteps to
            perform production MD for (4fs timestep).
        n_equil_cycles (int): Defaults to 3 cycles. Number of additional unrestrained 
            equilibration cycles to perform. Increasing this may triage unstable systems 
            at the cost of more simulation time albeit this may be negligible in the 
            grand scheme of things.
        reporter_frequency (int): Defaults to 1,000 timesteps. How often to write out to
            the logs and trajectory files in equilibration. X times less frequently during
            production MD.
        platform (str): Defaults to CUDA. Which OpenMM platform to simulate with (options
            include CUDA, CPU, OpenCL).
        device_ids (list[int]): Defaults to [0]. Which accelerators to use (multiple GPU 
            support is tenuous with OpenMM). Primarily used to distribute jobs to different
            GPUs on a node of an HPC resource.
        force_constant (float): Defaults to 10.0 kcal/mol*Å^2. Force constant to use for 
            harmonic restraints during equilibration. Currently restraints are only applied
            to protein backbone atoms.
    """
    def __init__(self, 
                 path: PathLike, 
                 equil_steps: int=1_250_000, 
                 prod_steps: int=250_000_000, 
                 n_equil_cycles: int=3,
                 reporter_frequency: int=1_000,
                 platform: str='CUDA',
                 device_ids: list[int]=[0],
                 force_constant: float=10.):
        self.path = Path(path) # enforce path object
        # input files
        self.prmtop = path / 'system.prmtop'
        self.inpcrd = path / 'system.inpcrd'

        # logging/checkpointing stuff
        self.eq_state = path / 'eq.state'
        self.eq_chkpt = path / 'eq.chk'
        self.eq_log = path / 'eq.log'
        self.eq_freq = reporter_frequency
        
        self.dcd = path / 'prod.dcd'
        self.restart = path / 'prod.rst.chk'
        self.state = path / 'prod.state'
        self.chkpt = path / 'prod.chk'
        self.prod_log = path / 'prod.log'
        self.prod_freq = self.eq_freq * 10

        # simulation details
        self.indices = self.get_restraint_indices()
        self.equil_cycles = n_equil_cycles
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.k = force_constant
        self.platform = Platform.getPlatformByName(platform)
        self.properties = {'DeviceIndex': ','.join([str(x) for x in device_ids]), 
                           'Precision': 'mixed'}

        if platform == 'CPU':
            self.properties = {}

    def load_amber_files(self) -> System:
        """
        Builds an OpenMM system using the prmtop/inpcrd files. PME is utilized for
        electrostatics and a 1 nm non-bonded cutoff is used as well as 1.5 amu HMR.

        Returns:
            (System): OpenMM system
        """
        if isinstance(self.inpcrd, Path | str):
            self.inpcrd = AmberInpcrdFile(str(self.inpcrd))
            self.prmtop = AmberPrmtopFile(str(self.prmtop), 
                                          periodicBoxVectors=self.inpcrd.boxVectors)

        system = self.prmtop.createSystem(nonbondedMethod=PME,
                                          removeCMMotion=False,
                                          nonbondedCutoff=1. * nanometer,
                                          constraints=HBonds,
                                          hydrogenMass=1.5 * amu)
    
        return system
    
    def setup_sim(self, 
                  system: System, 
                  dt: float) -> tuple[Simulation, Integrator]:
        """
        Builds OpenMM Integrator and Simulation objects utilizing a provided
        OpenMM System object and integration timestep, dt.

        Arguments:
            system (System): OpenMM system to build simulation object of.
            dt (float): Integration timestep in units of picoseconds.

        Returns:
            (tuple[Simulation, Integrator]): A tuple containing both the Simulation
                and Integrator objects.
        """
        integrator = LangevinMiddleIntegrator(300*kelvin, 
                                              1/picosecond, 
                                              dt*picoseconds)
        simulation = Simulation(self.prmtop.topology, 
                                system, 
                                integrator, 
                                self.platform, 
                                self.properties)
    
        return simulation, integrator

    def run(self) -> None:
        """
        Main logic of class. Determines whether or not we are restarting based on
        if all the equilibration outputs are present. Importantly the state file
        will not be written out until equilibration is complete. Also checks the
        production MD log to see if we have finished, otherwise decrements our
        progress from the total number of timesteps. Finally, runs production MD.
        """
        skip_eq = all([f.exists() 
                       for f in [self.eq_state, self.eq_chkpt, self.eq_log]])
        if not skip_eq:
            self.equilibrate()

        if self.restart.exists():
            self.check_num_steps_left()
            self.production(chkpt=self.restart, 
                            restart=True)
        else:
            self.production(chkpt=self.eq_chkpt,
                            restart=False)

    def equilibrate(self) -> Simulation:
        """
        Sets up and runs equilibrium MD, including energy minimization to begin.
        Sets backbone harmonic restraints and performs a slow heating protocol
        before periodically lessening restraints, finishing with user-specified
        number of rounds of unrestrained equilibration.

        Returns:
            (Simulation): OpenMM simulation object.
        """
        system = self.add_backbone_posres(self.load_amber_files(), 
                                          self.inpcrd.positions, 
                                          self.prmtop.topology.atoms(), 
                                          self.indices,
                                          self.k)
    
        simulation, integrator = self.setup_sim(system, dt=0.002)
        
        simulation.context.setPositions(self.inpcrd.positions)
        simulation.minimizeEnergy()
        
        simulation.reporters.append(StateDataReporter(str(self.eq_log), 
                                                      str(self.eq_freq), 
                                                      step=True,
                                                      potentialEnergy=True,
                                                      speed=True,
                                                      temperature=True))
        simulation.reporters.append(DCDReporter(str(self.path / 'eq.dcd'), self.eq_freq))

        simulation, integrator = self._heating(simulation, integrator)
        simulation = self._equilibrate(simulation)
        
        return simulation

    def production(self, 
                   chkpt: PathLike, 
                   restart: bool=False) -> None:
        """
        Performs production MD. Loads a new system, integrator and simulation object
        and loads equilibration or past production checkpoint to allow either the
        continuation of a past simulation or to inherit the outputs of equilibration.

        Arguments:
            chkpt (PathLike): The checkpoint file to load. Should be either the equilibration
                checkpoint or a past production checkpoint.
            restart (bool): Defaults to False. Flag to ensure we log the full simulation
                to reporter log. Otherwise restarts will overwrite the original log file.
        """
        system = self.load_amber_files()
        simulation, integrator = self.setup_sim(system, dt=0.004)
        
        system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
        simulation.context.reinitialize(True)

        if restart:
            log_file = open(str(self.prod_log), 'a')
        else:
            log_file = str(self.prod_log)

        simulation = self.load_checkpoint(simulation, chkpt)
        simulation = self.attach_reporters(simulation,
                                           self.dcd,
                                           log_file,
                                           self.restart,
                                           restart=restart)
    
        self._production(simulation)
    
    def load_checkpoint(self, 
                        simulation: Simulation, 
                        checkpoint: PathLike) -> Simulation:
        """
        Loads a previous checkpoint into provided simulation object.
        
        Arguments:
            simulation (Simulation): OpenMM simulation object.
            checkpoint (PathLike): OpenMM checkpoint file.

        Returns:
            (Simulation): OpenMM simulation object that has been checkpointed.
        """
        simulation.loadCheckpoint(checkpoint)
        state = simulation.context.getState(getVelocities=True, getPositions=True)
        positions = state.getPositions()
        velocities = state.getVelocities()
        
        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)

        return simulation

    def attach_reporters(self, 
                         simulation: Simulation, 
                         dcd_file: PathLike, 
                         log_file: PathLike, 
                         rst_file: PathLike, 
                         restart: bool=False) -> Simulation:
        """
        Attaches a StateDataReporter for logging, CheckpointReporter to output 
        periodic checkpoints and a DCDReporter to output trajectory data to 
        simulation object.

        Arguments:
            simulation (Simulation): OpenMM simulation object.
            dcd_file (PathLike): DCD file to write to.
            log_file (PathLike): Log file to write to.
            rst_file (PathLike): Checkpoint file to write to.
            restart (bool): Defaults to False. Whether or not we should be appending
                to an existing DCD file or writing a new one, potentially overwriting
                an existing DCD (if false).

        Returns:
            (Simulation): OpenMM simulation object with reporters now attached.
        """
        simulation.reporters.extend([
            DCDReporter(
                dcd_file, 
                self.prod_freq,
                append=restart
                ),
            StateDataReporter(
                log_file,
                self.prod_freq,
                step=True,
                potentialEnergy=True,
                temperature=True,
                progress=True,
                remainingTime=True,
                speed=True,
                volume=True,
                totalSteps=self.prod_steps,
                separator='\t'
                ),
            CheckpointReporter(
                rst_file,
                self.prod_freq * 10
                )
            ])

        return simulation

    def _heating(self, 
                 simulation: Simulation, 
                 integrator: Integrator) -> tuple[Simulation, Integrator]:
        """
        Slow heating protocol. Seeds velocities at 5 kelvin to begin, and
        over a period of 100,000 timesteps gradually heats up to 300 kelvin
        in 1,000 discrete jumps. This should generalize to most systems but
        if your system requires more control these hard-coded values will need
        to be changed here.

        Arguments:
            simulation (Simulation): OpenMM simulation object.
            integrator (Integrator): OpenMM integrator object.

        Returns:
            (tuple[Simulation, Integrator]): Tuple containing simulation and
                integrator objects after heating has been performed.
        """
        simulation.context.setVelocitiesToTemperature(5*kelvin)
        T = 5
        
        integrator.setTemperature(T * kelvin)
        mdsteps = 100000
        length = mdsteps // 1000
        tstep = (300 - T) / length
        for i in range(length):
          simulation.step(mdsteps//60)
          temp = T + tstep * (1 + i)
          
          if temp > 300:
            temp = 300
          
          integrator.setTemperature(temp * kelvin)
    
        return simulation, integrator
         
    def _equilibrate(self, 
                     simulation: Simulation) -> Simulation:
        """
        Equilibration procotol. 
        (1) Performs a 5-step restraint relaxation in NVT ensemble wherein 
            each step 1/5 of the restraint is relaxed until there are no harmonic 
            restraints. Length of each step is also 1/5 of the total equil_steps 
            set by the user. 
        (2) One step-length of unrestrained NVT is performed before turning on 
            barostat for NPT.
        (3) `equil_cycles` number of step-lengths are then ran with NPT before
            saving out the state and checkpoints.

        Arguments:
            simulation (Simulation): OpenMM simulation object.

        Returns:
            (Simulation): Equilibrated OpenMM simulation object.
        """
        simulation.context.reinitialize(True)
        n_levels = 5
        d_k = self.k / n_levels
        eq_steps = self.equil_steps // n_levels

        for i in range(n_levels): 
            simulation.step(eq_steps)
            k = float(self.k - (i * d_k))
            simulation.context.setParameter('k', (k * kilocalories_per_mole/angstroms**2))
        
        simulation.context.setParameter('k', 0)
        simulation.step(eq_steps)
    
        simulation.system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
        simulation.step(self.equil_cycles * eq_steps)

        simulation.saveState(str(self.eq_state))
        simulation.saveCheckpoint(str(self.eq_chkpt))
    
        return simulation
    
    def _production(self, 
                    simulation: Simulation) -> Simulation:
        """
        Production MD procotol. Runs for specified number of timesteps before
        saving out a final state and checkpoint file.

        Arguments:
            simulation (Simulation): OpenMM simulation object.

        Returns:
            (Simulation): OpenMM simulation object.
        """
        simulation.step(self.prod_steps)
        simulation.saveState(str(self.state))
        simulation.saveCheckpoint(str(self.chkpt))
    
        return simulation

    def get_restraint_indices(self, 
                              addtl_selection: str='') -> list[int]:
        """
        Obtains atom indices that will be used to set harmonic restraints. First
        loads an MDAnalysis universe with the input prmtop and inpcrd files. Uses
        a base selection of protein or nucleic acid backbone but if provided an
        additional selection can be included for things like restraining ligand
        molecules.

        Arguments:
            addtl_selection (str): Defaults to empty string. If provided, will
                be used to define additional atoms for restraining.

        Returns:
            (list[int]): List of atomic indices for atoms to be restrained.
        """
        u = mda.Universe(str(self.prmtop), str(self.inpcrd))
        if addtl_selection:
            sel = u.select_atoms(f'backbone or nucleicbackbone or {addtl_selection}')
        else:
            sel = u.select_atoms('backbone or nucleicbackbone')
            
        return sel.atoms.ix
        
    def check_num_steps_left(self) -> None:
        """
        Reads the production log file to see if we have completed a simulation.
        If there is still simulation to be completed, decrements the existing progress
        from internal number of steps to perform. Additionally, accounts for any frames
        that have been written to DCD that are not accounted for in the log.

        Returns:
            None
        """
        prod_log = open(str(self.prod_log)).readlines()

        try:
            last_line = prod_log[-1]
            last_step = int(last_line.split()[1].strip())
        except IndexError:
            try:
                last_line = prod_log[-2]
                last_step = int(last_line.split()[1].strip())
            except IndexError: # something weird happend just run full time
                return
        
        if time_left := (self.prod_steps - last_step):
            self.prod_steps -= time_left

            if n_repeat_timesteps := (last_step % (self.prod_freq * 10)):
                self.prod_steps -= n_repeat_timesteps
                n_repeat_frames = n_repeat_timesteps / self.prod_freq
                
                n_total_frames = last_step / self.prod_freq
                
                lines = [f'{n_total_frames - n_repeat_frames},{n_total_frames}']
                duplicate_log = self.path / 'duplicate_frames.log'
                if duplicate_log.exists():
                    mode = 'a'
                else:
                    mode = 'w'
                    lines = ['first_frame,last_frame'] + lines
                    
                with open(str(duplicate_log), mode) as fout:
                    fout.write('\n'.join(lines))

    @staticmethod
    def add_backbone_posres(system: System, 
                            positions: np.ndarray, 
                            atoms: list[topology.Atom], 
                            indices: list[int], 
                            restraint_force: float=10.) -> System:
        """
        Adds harmonic restraints to an OpenMM system.

        Arguments:
            system (System): OpenMM system object.
            positions (np.ndarray): Position array for all atoms in system.
            atoms (list[topology.Atom]): List of all Atom objects in system.
            indices (list[int]): List of atomic indices to restrain.
            restraint_force (float): Defaults to 10.0 kcal/mol*Å^2. The force
                constant to use for harmonic restraints.

        Returns:
            (System): OpenMM system with harmonic restraints.
        """
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    
        force_amount = restraint_force * kilocalories_per_mole/angstroms**2
        force.addGlobalParameter("k", force_amount)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
    
        for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
            if atom.index in indices:
                force.addParticle(i, atom_crd.value_in_unit(nanometers))
      
        posres_sys = deepcopy(system)
        posres_sys.addForce(force)
      
        return posres_sys

class ImplicitSimulator(Simulator):
    """
    Implicit solvent simulator. Inherits from Simulator and overloads relevant
    methods to support the slight differences in how implicit solvent is implemented
    in OpenMM.
    
    Arguments:
        path (PathLike): Path to simulation inputs, same as output path.
        equil_steps (int): Defaults to 1,250,000 (2.5 ns). Number of simulation timesteps to
            perform equilibration for (2fs timestep).
        prod_steps (int): Defaults to 250,000,000 (1 µs). Number of simulation timesteps to
            perform production MD for (4fs timestep).
        n_equil_cycles (int): Defaults to 3 cycles. Number of additional unrestrained 
            equilibration cycles to perform. Increasing this may triage unstable systems 
            at the cost of more simulation time albeit this may be negligible in the 
            grand scheme of things.
        reporter_frequency (int): Defaults to 1,000 timesteps. How often to write out to
            the logs and trajectory files in equilibration. X times less frequently during
            production MD.
        platform (str): Defaults to CUDA. Which OpenMM platform to simulate with (options
            include CUDA, CPU, OpenCL).
        device_ids (list[int]): Defaults to [0]. Which accelerators to use (multiple GPU 
            support is tenuous with OpenMM). Primarily used to distribute jobs to different
            GPUs on a node of an HPC resource.
        force_constant (float): Defaults to 10.0 kcal/mol*Å^2. Force constant to use for 
            harmonic restraints during equilibration. Currently restraints are only applied
            to protein backbone atoms.
        implicit_solvent (Singleton): Defaults to GBn2. Which implicit solvent model to use.
        solute_dielectric (float): Defaults to 1.0. Probably shouldn't change this.
        solvent_dielectric (float): Defaults to 78.5. Also shouldn't change this.
    """
    def __init__(self, 
                 path: str, 
                 equil_steps: int=1_250_000, 
                 prod_steps: int=250_000_000, 
                 n_equil_cycles: int=3,
                 reporter_frequency: int=1_000,
                 platform: str='CUDA',
                 device_ids: list[int]=[0],
                 force_constant: float=10.,
                 implicit_solvent: Singleton=GBn2,
                 solute_dielectric: float=1.,
                 solvent_dielectric: float=78.5):
        super().__init__(path, equil_steps, prod_steps, n_equil_cycles,
                         reporter_frequency, platform, device_ids, 
                         force_constant)
        self.solvent = implicit_solvent
        self.solute_dielectric = solute_dielectric
        self.solvent_dielectric = solvent_dielectric
        # solvent screening parameter for 150mM ions
        # k = 367.434915 * sqrt(conc. [M] / (solvent_dielectric * T))
        self.kappa = 367.434915 * np.sqrt(.15 / (solvent_dielectric * 300))
    
    def load_amber_files(self) -> System:
        """
        Loads an OpenMM system object with implicit solvent parameters.

        Returns:
            (System): OpenMM system object.
        """
        if isinstance(self.inpcrd, Path | str):
            self.inpcrd = AmberInpcrdFile(self.inpcrd)
            self.prmtop = AmberPrmtopFile(self.prmtop)

        system = self.prmtop.createSystem(nonbondedMethod=NoCutoff,
                                          removeCMMotion=False,
                                          constraints=HBonds,
                                          hydrogenMass=1.5 * amu,
                                          implicitSolvent=self.solvent,
                                          soluteDielectric=self.solute_dielectric,
                                          solventDielectric=self.solvent_dielectric,
                                          implicitSolventKappa=self.kappa/nanometer)
    
        return system
        
    def equilibrate(self) -> Simulation:
        """
        Runs reduced equilibration protocol. Due to the faster convergence of
        using implicit solvent we don't need to be quite as rigorous as with
        explicit solvent system relaxation.

        Returns:
            (Simulation): OpenMM simulation object.
        """
        system = self.add_backbone_posres(self.load_amber_files(), 
                                          self.inpcrd.positions, 
                                          self.prmtop.topology.atoms(), 
                                          self.indices,
                                          self.k)
    
        simulation, integrator = self.setup_sim(system, dt=0.002)
        
        simulation.context.setPositions(self.inpcrd.positions)
        state = simulation.context.getState(getEnergy=True)
        print(f'Energy before minimization: {state.getPotentialEnergy()}')
        simulation.minimizeEnergy()
        state = simulation.context.getState(getEnergy=True)
        print(f'Energy after minimization: {state.getPotentialEnergy()}')
        
        simulation.reporters.append(StateDataReporter(self.eq_log, 
                                                      self.eq_freq, 
                                                      step=True,
                                                      potentialEnergy=True,
                                                      speed=True,
                                                      temperature=True))
        simulation.reporters.append(DCDReporter(f'{self.path}/eq.dcd', self.eq_freq))

        simulation, integrator = self._heating(simulation, integrator)
        simulation = self._equilibrate(simulation)
        
        return simulation

class CustomForcesSimulator(Simulator):
    """
    Simulator for utilizing custom user-defined forces. Inherits from Simulator while
    providing a way to inject custom forces.

    Arguments:
        path (PathLike): Path to simulation inputs, same as output path.
        custom_force_objects (list): ??
        equil_steps (int): Defaults to 1,250,000 (2.5 ns). Number of simulation timesteps to
            perform equilibration for (2fs timestep).
        prod_steps (int): Defaults to 250,000,000 (1 µs). Number of simulation timesteps to
            perform production MD for (4fs timestep).
        n_equil_cycles (int): Defaults to 3 cycles. Number of additional unrestrained 
            equilibration cycles to perform. Increasing this may triage unstable systems 
            at the cost of more simulation time albeit this may be negligible in the 
            grand scheme of things.
        reporter_frequency (int): Defaults to 1,000 timesteps. How often to write out to
            the logs and trajectory files in equilibration. X times less frequently during
            production MD.
        platform (str): Defaults to CUDA. Which OpenMM platform to simulate with (options
            include CUDA, CPU, OpenCL).
        device_ids (list[int]): Defaults to [0]. Which accelerators to use (multiple GPU 
            support is tenuous with OpenMM). Primarily used to distribute jobs to different
            GPUs on a node of an HPC resource.
        force_constant (float): Defaults to 10.0 kcal/mol*Å^2. Force constant to use for 
            harmonic restraints during equilibration. Currently restraints are only applied
            to protein backbone atoms.
    """
    def __init__(self,
                 path: str,
                 custom_force_objects: list,
                 equil_steps: int=1_250_000, 
                 prod_steps: int=250_000_000, 
                 n_equil_cycles: int=3,
                 reporter_frequency: int=1_000,
                 platform: str='CUDA',
                 device_ids: list[int]=[0],
                 equilibration_force_constant: float=10.):
        super().__init__(path, equil_steps, prod_steps, n_equil_cycles,
                         reporter_frequency, platform, device_ids, 
                         equilibration_force_constant)
        self.custom_forces = custom_force_objects

    def load_amber_files(self) -> System:
        """
        Loads OpenMM system and adds in custom forces.

        Returns:
            (System): OpenMM system object with custom forces.
        """
        if isinstance(self.inpcrd, str):
            self.inpcrd = AmberInpcrdFile(self.inpcrd)
            self.prmtop = AmberPrmtopFile(self.prmtop, periodicBoxVectors=self.inpcrd.boxVectors)

        system = self.prmtop.createSystem(nonbondedMethod=PME,
                                          removeCMMotion=False,
                                          nonbondedCutoff=1. * nanometer,
                                          constraints=HBonds,
                                          hydrogenMass=1.5 * amu)

        system = self.add_forces(system)

        return system

    def add_forces(self, 
                   system: System) -> System:
        """
        Systematically adds all custom forces to provided system.

        Arguments:
            system (System): OpenMM system object.

        Returns:
            (System): OpenMM system object with custom forces added.
        """
        for custom_force in self.custom_forces:
            system.addForce(custom_force)

        return system

class Minimizer:
    """
    Class for just performing energy minimization and writing out minimized structures.

    Arguments:
        topology (PathLike): Topology file which can be either a PDB or an AMBER prmtop.
        coordinates (OptPath): Defaults to None. If using a prmtop, you need to provide
            an inpcrd file for the coordinates. Otherwise these are also acquired from
            the PDB used as topology.
        out (PathLike): Defaults to min.pdb. The name of the output minimized PDB file.
            The path is inherited from the parent path of the topology file.
        platform (str): Defaults to OpenCL. Which OpenMM platform to run.
        device_ids (list[int]): Accelerator IDs.
    """
    def __init__(self,
                 topology: PathLike,
                 coordinates: OptPath = None,
                 out: PathLike='min.pdb',
                 platform: str='OpenCL',
                 device_ids: list[int]=[0]):
        self.topology = Path(topology) 
        if coordinates is not None:
            self.coordinates = Path(coordinates)

        self.path = self.topology.parent
        self.out = self.path / out
        self.platform = platform
        self.properties = {'DeviceIndex': ','.join([str(x) for x in device_ids]),
                           'Precision': 'mixed'}

    def minimize(self) -> None:
        """
        Loads an OpenMM system, builds simulation object and runs energy minimization.
        Dumps final coordinates into output PDB file.

        Returns:
            None
        """
        system = self.load_files()
        integrator = LangevinMiddleIntegrator(300*kelvin, 
                                              1/picosecond, 
                                              0.001*picoseconds)
        simulation = Simulation(self.topology, 
                                system, 
                                integrator,
                                self.platform,
                                self.properties)

        simulation.context.setPositions(self.coordinates.positions)

        simulation.minimizeEnergy()

        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        
        PDBFile.writeFile(simulation.topology, 
                          positions, 
                          file=str(self.out), 
                          keepIds=True)

    def load_files(self) -> None:
        """
        Loads an OpenMM system depending on what file types you have provided
        for the topology (AMBER, PDB, etc).

        Returns:
            None
        """
        if self.topology.suffix in ['.prmtop', '.parm7']:
            system = self.load_amber()
        elif self.topology.suffix == '.top':
            system = self.load_gromacs()
        elif self.topology.suffix == '.pdb':
            system = self.load_pdb()
        else:
            raise FileNotFoundError('No viable simulation input files found'
                                    f'at path: {self.path}!')

        return system
        
    def load_amber(self) -> System:
        """
        Loads AMBER input files into OpenMM System.

        Returns:
            (System): OpenMM system object.
        """
        inpcrd = list(self.path.glob('*.inpcrd')) + list(self.path.glob('*.rst7'))
        if hasattr(self, 'coordinates'):
            self.coordinates = AmberInpcrdFile(str(self.coordinates))
        else:
            self.coordinates = AmberInpcrdFile(str(inpcrd[0]))
        self.topology = AmberPrmtopFile(str(self.topology))

        system = self.topology.createSystem(nonbondedMethod=NoCutoff,
                                            constraints=HBonds)

        return system

    def load_gromacs(self) -> System:
        """
        Loads GROMACS input files into OpenMM System. UNTESTED!

        Returns:
            (System): OpenMM system object.
        """
        gro = list(self.path.glob('*.gro'))[0]
        self.coordinates = GromacsGroFile(str(gro))
        self.topology = GromacsTopFile(str(self.topology), 
                                       includeDir='/usr/local/gromacs/share/gromacs/top')

        system = self.topology.createSystem(nonbondedMethod=NoCutoff, 
                                            constraints=HBonds)

        return system

    def load_pdb(self) -> System:
        """
        Loads PDB into OpenMM System. Beware the topology guesser.

        Returns:
            (System): OpenMM system object.
        """
        self.coordinates = PDBFile(str(self.topology))
        self.topology = self.coordinates.topology
        forcefield = ForceField('amber14-all.xml')

        system = forcefield.createSystem(self.topology, 
                                         nonbondedMethod=NoCutoff,
                                         constraints=HBonds)

        return system
