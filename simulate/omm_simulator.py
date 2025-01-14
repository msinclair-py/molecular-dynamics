from copy import deepcopy
from openmm import *
from openmm.app import *
from openmm.unit import *

class Simulator:
    def __init__(self, path, equil_steps: int=500_000, prod_steps: int=250_000_000):
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
        self.platform = Platform.getPlatformByName('CUDA')
        self.properties = {'Precision': 'mixed'}

    def load_amber_files(self):
        self.inpcrd = AmberInpcrdFile(self.inpcrd)
        self.prmtop = AmberPrmtopFile(self.prmtop, periodicBoxVectors=self.inpcrd.boxVectors)
        system = self.prmtop.createSystem(nonbondedMethod=PME,
                                          removeCMMotion=False,
                                          nonbondedCutoff=1. * nanometer,
                                          constraints=HBonds,
                                          hydrogenMass=1.5 * amu)
    
        return system
    
    def setup_sim(self, system):
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(self.prmtop.topology, system, integrator, self.platform, self.properties)
    
        return simulation, integrator

    def equilibrate(self):
        system = self.add_backbone_posres(self.load_amber_files(), 
                                          self.inpcrd.positions, 
                                          self.prmtop.topology.atoms(), 
                                          10)
    
        simulation, integrator = self.setup_sim(system)
        
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

    def production(self):
        system = self.load_amber_files()
        simulation, integrator = self.setup_sim(system)
        
        system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
        simulation.context.reinitialize(True)
    
        self._production(simulation)
    
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
    
        for i in range(100):
            simulation.step(self.equil_steps // 100)
            k = float(99.02 - (i * 0.98))
            simulation.context.setParameter('k', (k * kilocalories_per_mole/angstroms**2))
        
        simulation.context.setParameter('k', 0)
        simulation.saveState(self.eq_state)
        simulation.saveCheckpoint(self.eq_chkpt)
    
        return simulation
    
    def _production(self, simulation):
    
        simulation.loadCheckpoint(self.eq_chkpt)
        eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
        positions = eq_state.getPositions()
        velocities = eq_state.getVelocities()
        
        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)
        
        simulation.reporters.extend([
            DCDReporter(
                self.dcd, 
                10000
                ),
            StateDataReporter(
                self.prod_log,
                10000,
                step=True,
                potentialEnergy=True,
                temperature=True,
                progress=True,
                remainingTime=True,
                speed=True,
                volume=True,
                totalSteps=md_steps,
                separator='\t'
                ),
            CheckpointReporter(
                self.restart,
                100000
                )
            ])
    
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

