from copy import deepcopy
import MDAnalysis as mda
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.app.internal.singleton import Singleton
import os
from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[Path, str]

class Simulator:
    def __init__(self, 
                 path: PathLike, 
                 equil_steps: int=1_250_000, 
                 prod_steps: int=250_000_000, 
                 n_equil_cycles: int=3,
                 reporter_frequency: int=1_000,
                 platform: str='CUDA',
                 device_ids: list[int]=[0],
                 force_constant: float=10.):
        self.path = path if isinstance(path, Path) else Path(path)
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
        if isinstance(self.inpcrd, Path | str):
            self.inpcrd = AmberInpcrdFile(str(self.inpcrd))
            self.prmtop = AmberPrmtopFile(str(self.prmtop), periodicBoxVectors=self.inpcrd.boxVectors)

        system = self.prmtop.createSystem(nonbondedMethod=PME,
                                          removeCMMotion=False,
                                          nonbondedCutoff=1. * nanometer,
                                          constraints=HBonds,
                                          hydrogenMass=1.5 * amu)
    
        return system
    
    def setup_sim(self, system: System, dt: float) -> Tuple[Simulation, Integrator]:
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

    def production(self, chkpt: str, restart: bool=False) -> None:
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
                        checkpoint: str) -> Simulation:
        simulation.loadCheckpoint(checkpoint)
        state = simulation.context.getState(getVelocities=True, getPositions=True)
        positions = state.getPositions()
        velocities = state.getVelocities()
        
        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)

        return simulation

    def attach_reporters(self, 
                         simulation: Simulation, 
                         dcd_file: str, 
                         log_file: str, 
                         rst_file: str, 
                         restart: bool=False) -> Simulation:
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
                 integrator: Integrator) -> Tuple[Simulation, Integrator]:
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
         
    def _equilibrate(self, simulation: Simulation) -> Simulation:
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

        simulation.system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
        simulation.step(3*eq_steps)

        simulation.saveState(str(self.eq_state))
        simulation.saveCheckpoint(str(self.eq_chkpt))
    
        return simulation
    
    def _production(self, simulation: Simulation) -> Simulation:
        simulation.step(self.prod_steps)
        simulation.saveState(str(self.state))
        simulation.saveCheckpoint(str(self.chkpt))
    
        return simulation

    def get_restraint_indices(self, addtl_selection: str='') -> List[int]:
        u = mda.Universe(str(self.prmtop), str(self.inpcrd))
        if addtl_selection:
            sel = u.select_atoms(f'backbone or nucleicbackbone or {addtl_selection}')
        else:
            sel = u.select_atoms('backbone or nucleicbackbone')
            
        return sel.atoms.ix
        
    def check_num_steps_left(self) -> None:
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
    def add_backbone_posres(system: System, positions: np.ndarray, atoms: List[str], 
                            indices: List[int], restraint_force: float=10.) -> System:
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
        # k = 367.434915 * sqrt(conc. [M] / (solvent_dielectric * T))
        self.kappa = 367.434915 * np.sqrt(.15 / (solvent_dielectric * 300))
    
    def load_amber_files(self) -> System:
        if isinstance(self.inpcrd, str):
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
        if isinstance(self.inpcrd, str):
            self.inpcrd = AmberInpcrdFile(self.inpcrd)
            try: # This is how it is done in OpenMM 8.0 and on
                self.prmtop = AmberPrmtopFile(self.prmtop, periodicBoxVectors=self.inpcrd.boxVectors)
            except TypeError: # This means we are in OpenMM 7.7 or earlier
                self.prmtop = AmberPrmtopFile(self.prmtop)

        system = self.prmtop.createSystem(nonbondedMethod=PME,
                                          removeCMMotion=False,
                                          nonbondedCutoff=1. * nanometer,
                                          constraints=HBonds,
                                          hydrogenMass=1.5 * amu)

        system = self.add_forces(system)

        return system

    def add_forces(self, system: System) -> System:
        for custom_force in self.custom_forces:
            system.addForce(custom_force)

        return system

class Minimizer:
    def __init__(self,
                 topology: PathLike,
                 out: PathLike='min.pdb',
                 platform: str='OpenCL',
                 device_ids: list[int]=[0]):
        self.topology = Path(topology) if isinstance(topology, str) else topology
        self.path = self.topology.parent
        self.out = self.path / out
        self.platform = platform
        self.properties = {'DeviceIndex': ','.join([str(x) for x in device_ids]),
                           'Precision': 'mixed'}

    def minimize(self) -> None:
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
        inpcrd = list(self.path.glob('*.inpcrd')) + list(self.path.glob('*.rst7'))
        self.coordinates = AmberInpcrdFile(str(inpcrd[0]))
        self.topology = AmberPrmtopFile(str(self.topology))

        system = self.topology.createSystem(nonbondedMethod=NoCutoff,
                                            constraints=HBonds)

        return system

    def load_gromacs(self) -> System:
        gro = list(self.path.glob('*.gro'))[0]
        self.coordinates = GromacsGroFile(str(gro))
        self.topology = GromacsTopFile(str(self.topology), 
                                       includeDir='/usr/local/gromacs/share/gromacs/top')

        system = self.topology.createSystem(nonbondedMethod=NoCutoff, 
                                            constraints=HBonds)

        return system

    def load_pdb(self) -> System:
        self.coordinates = PDBFile(str(self.topology))
        self.topology = self.coordinates.topology
        forcefield = ForceField('amber14-all.xml')

        system = forcefield.createSystem(self.topology, 
                                         nonbondedMethod=NoCutoff,
                                         constraints=HBonds)

        return system
