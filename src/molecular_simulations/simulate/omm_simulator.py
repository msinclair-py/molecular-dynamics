from copy import deepcopy
from openmm import *
from openmm.app import *
from openmm.unit import *
import os

class Simulator:
    def __init__(self, path: str, equil_steps: int=1_250_000, 
                 prod_steps: int=250_000_000, force_constant: float=10.):
        self.path = path
        # input files
        self.prmtop = f'{path}/system.prmtop'
        self.inpcrd = f'{path}/system.inpcrd'

        # logging/checkpointing stuff
        self.eq_state = f'{path}/eq.state'
        self.eq_chkpt = f'{path}/eq.chk'
        self.eq_log = f'{path}/eq.log'
        
        self.dcd = f'{path}/prod.dcd'
        self.restart = f'{path}/prod.rst.chk'
        self.state = f'{path}/prod.state'
        self.chkpt = f'{path}/prod.chk'
        self.prod_log = f'{path}/prod.log'

        # simulation details
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.k = force_constant
        self.platform = Platform.getPlatformByName('CUDA')
        self.properties = {'Precision': 'mixed'}

    def load_amber_files(self):
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
    
        return system
    
    def setup_sim(self, system, dt):
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, dt*picoseconds)
        simulation = Simulation(self.prmtop.topology, system, integrator, self.platform, self.properties)
    
        return simulation, integrator

    def run(self):
        skip_eq = all([os.path.exists(f) for f in [self.eq_state, self.eq_chkpt, self.eq_log]])
        reload_prod = os.path.exists(self.restart)
        if not skip_eq:
            self.equilibrate()

        if reload_prod:
            self.production(chkpt=self.restart, 
                            restart=True)
        else:
            self.production(chkpt=self.eq_chkpt,
                            restart=False)

    def equilibrate(self):
        system = self.add_backbone_posres(self.load_amber_files(), 
                                          self.inpcrd.positions, 
                                          self.prmtop.topology.atoms(), 
                                          self.k)
    
        simulation, integrator = self.setup_sim(system, dt=0.002)
        
        # OpenMM 7.7 requires us to do this. Redundant but harmless if we are
        # in version 8.1 or on
        simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        
        simulation.context.setPositions(self.inpcrd.positions)
        simulation.minimizeEnergy()
        
        simulation.reporters.append(StateDataReporter(self.eq_log, 
                                                      1000, 
                                                      step=True,
                                                      potentialEnergy=True,
                                                      speed=True,
                                                      temperature=True))

        simulation, integrator = self._heating(simulation, integrator)
        simulation = self._equilibrate(simulation)
        
        return simulation

    def production(self, chkpt, restart):
        system = self.load_amber_files()
        simulation, integrator = self.setup_sim(system, dt=0.004)
        
        system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
        simulation.context.reinitialize(True)

        if restart:
            log_file = open(self.prod_log, 'a')
        else:
            log_file = self.prod_log

        simulation = self.load_checkpoint(simulation, chkpt)
        simulation = self.attach_reporters(simulation,
                                           self.dcd,
                                           log_file,
                                           self.restart,
                                           restart=restart)
    
        self._production(simulation)
    
    def load_checkpoint(self, simulation, checkpoint):
        simulation.loadCheckpoint(checkpoint)
        state = simulation.context.getState(getVelocities=True, getPositions=True)
        positions = state.getPositions()
        velocities = state.getVelocities()
        
        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)

        return simulation

    def attach_reporters(self, simulation, dcd_file, log_file, rst_file, restart=False):
        simulation.reporters.extend([
            DCDReporter(
                dcd_file, 
                10000,
                append=restart
                ),
            StateDataReporter(
                log_file,
                10000,
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
                100000
                )
            ])

        return simulation

    def _heating(self, simulation, integrator):
        simulation.context.setVelocitiesToTemperature(5*kelvin)
        T = 5
        
        integrator.setTemperature(T * kelvin)
        mdsteps = 60000
        for i in range(60):
          simulation.step(mdsteps//60)
          temp = T * (1 + i)
          
          if temp > 300:
            temp = 300
          
          integrator.setTemperature(temp * kelvin)
    
        return simulation, integrator
         
    def _equilibrate(self, simulation):
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
        simulation.step(3*eq_steps)

        simulation.saveState(self.eq_state)
        simulation.saveCheckpoint(self.eq_chkpt)
    
        return simulation
    
    def _production(self, simulation):
        simulation.step(self.prod_steps)
        simulation.saveState(self.state)
        simulation.saveCheckpoint(self.chkpt)
    
        return simulation

    @staticmethod
    def add_backbone_posres(system, positions, atoms,
                            restraint_force):
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    
        force_amount = restraint_force * kilocalories_per_mole/angstroms**2
        force.addGlobalParameter("k", force_amount)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
    
        for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
            if atom.name in  ('CA', 'C', 'N', 'O'):
                force.addParticle(i, atom_crd.value_in_unit(nanometers))
      
        posres_sys = deepcopy(system)
        posres_sys.addForce(force)
      
        return posres_sys

