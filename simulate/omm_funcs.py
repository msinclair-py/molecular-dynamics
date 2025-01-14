from copy import deepcopy
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
import pandas as pd
from parmed import load_file, unit as u
from simtk import unit

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

def set_pbc(modeller):
    coords = modeller.positions
    min_crds = [coords[0][0], coords[0][1], coords[0][2]]
    max_crds = [coords[0][0], coords[0][1], coords[0][2]]
    
    for coord in coords:
        min_crds[0] = min(min_crds[0], coord[0])
        min_crds[1] = min(min_crds[1], coord[1])
        min_crds[2] = min(min_crds[2], coord[2])
        max_crds[0] = max(max_crds[0], coord[0])
        max_crds[1] = max(max_crds[1], coord[1])
        max_crds[2] = max(max_crds[2], coord[2])
    
    system.setPeriodicBoxVectors(max_crds[0]-min_crds[0],
                     max_crds[1]-min_crds[1],
                     max_crds[2]-min_crds[2],
    )
    return modeller

def load_amber_files(inpcrd_fil, prmtop_fil):
    inpcrd = AmberInpcrdFile(inpcrd_fil)
    prmtop = AmberPrmtopFile(prmtop_fil, periodicBoxVectors=inpcrd.boxVectors)
    system = prmtop.createSystem(nonbondedMethod=PME,
                                removeCMMotion=False,
                                nonbondedCutoff=1*nanometer,
                                constraints=HBonds,
                                hydrogenMass=1.5*amu)

    return system, prmtop, inpcrd

def system_implicit(input_pdb):
    pdb = PDBFile(input_pdb)
    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    system = forcefield.createSystem(pdb.topology,
                                    soluteDielectric=1.0,
                                    solventDielectric=80.0,
                                    hydrogenMass=1.5*amu,
                                    constraints=HBonds, 
                                    implicitSolventKappa=1.0/nanometer)
    return system, pdb, forcefield

def sim_implicit(system,
                pdb,
                simulation_time, 
                output_dcd, 
                output_pdb,
                out_log):

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}

    simulation = Simulation(pdb.topology,
                            system,
                            integrator,
                            platform,
                            properties)
    
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    
    simulation.reporters.append(StateDataReporter(
                                out_log,
                                10000,
                                potentialEnergy=True,
                                speed=True,
                                ))

    simulation.reporters.append(DCDReporter(output_dcd,
                                             25000))

    simulation.reporters.append(PDBReporter(output_pdb,
                                             simulation_time))

    simulation.step(simulation_time)
    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    return simulation, potential_energy.value_in_unit(unit.kilocalories_per_mole)

def setup_sim_nomin(system, prmtop, inpcrd, log):
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}

    simulation = Simulation(prmtop.topology,
                            system,
                            integrator,
                            platform,
                            properties)

    simulation.reporters.append(StateDataReporter(log,
                                10000,
                                step=True,
                                potentialEnergy=True,
                                speed=True,
                                temperature=True))

    return simulation, integrator


def setup_sim(system, prmtop, inpcrd, log):
    posres_sys = add_backbone_posres(system, inpcrd.positions, prmtop.topology.atoms(), 10)
    integrator = LangevinMiddleIntegrator(5*kelvin, 1/picosecond, 0.004*picoseconds)

    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}

    simulation = Simulation(prmtop.topology,
                            posres_sys,
                            integrator,
                            platform,
                            properties)
    
    simulation.context.setPositions(inpcrd.positions)
    simulation.minimizeEnergy()

    simulation.reporters.append(StateDataReporter(log,
                                1000,
                                step=True,
                                potentialEnergy=True,
                                speed=True,
                                temperature=True))
    
    simulation.step(10000)

    return posres_sys, simulation, integrator

def warming(simulation, integrator):
    simulation.context.setVelocitiesToTemperature(5*kelvin)
    print('Warming up the system...')
    T = 5
    mdsteps = 60000
    for i in range(60):
      simulation.step(int(mdsteps/60) )
      temp = (T+(i*T))
      if temp>300:
        temp = 300
      temperature = temp*kelvin 
      integrator.setTemperature(temperature)
    return simulation, integrator
     

def equilib(simulation,
            mdsteps,
            chkpt,
            state_out):
    simulation.context.reinitialize(True)

    for i in range(100):
        simulation.step(int(mdsteps/100))
        k = float(99.02-(i*0.98))
        simulation.context.setParameter('k', (k * kilocalories_per_mole/angstroms**2))
    
    simulation.context.setParameter('k', 0)
    simulation.saveState(state_out)
    simulation.saveCheckpoint(chkpt)

    return simulation


def run_eq(inpcrd_fil,
        prmtop_fil,
        state_out,
        chkpt,
        log_file,
        mdsteps=500000):

    system, prmtop, inpcrd = load_amber_files(inpcrd_fil,
                                            prmtop_fil)

    posres_sys, simulation, integrator = setup_sim(system,
                                                   prmtop,
                                                   inpcrd,
                                                   log_file)

    simulation, integrator = warming(simulation, integrator)

    simulation = equilib(simulation, mdsteps, chkpt, state_out)
    
    return simulation

def run_prod(simulation, 
            md_steps,
            chkpt,
            output_dcd,
            rst_chk,
            out_chk,
            out_st,
            out_log):

    simulation.loadCheckpoint(chkpt)
    eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
    positions = eq_state.getPositions()
    velocities = eq_state.getVelocities()
    
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)
    
    simulation.reporters.append(
        DCDReporter(output_dcd, 10000))
    
    simulation.reporters.append(
        StateDataReporter(
            out_log,
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
            )
        )
    
    simulation.reporters.append(
        CheckpointReporter(
            rst_chk,
            100000
            )
        )

    print('Running Production...')
    simulation.step(md_steps)
    simulation.saveState(out_st)
    simulation.saveCheckpoint(out_chk)
    return simulation

def prod(inpcrd, prmtop, eq_chkpt, traj, rst, chkpt, state, log, num_steps=1000000):
    system, prmtop, inpcrd = load_amber_files(inpcrd, prmtop)
    path, _ = os.path.split(eq_chkpt)
    eq_log = f'{path}/eq.log'
    eq_simulation, integrator = setup_sim_nomin(system, prmtop, inpcrd, eq_log)
    
    system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
    eq_simulation.context.reinitialize(True)

    run_prod(eq_simulation, num_steps, eq_chkpt, traj, rst, chkpt, state, log)

def run_md(path: str, n_steps=1_000_000):
    inpcrd = f'{path}/system.inpcrd'
    prmtop = f'{path}/system.prmtop'
    
    eq_state = f'{path}/eq.state'
    eq_chkpt = f'{path}/eq.chk'
    eq_log = f'{path}/eq.log'
    run_eq(inpcrd, prmtop, eq_state, eq_chkpt, eq_log)

    traj = f'{path}/prod.dcd'
    rst = f'{path}/prod.rst.chk'
    chkpt = f'{path}/prod.chk'
    state = f'{path}/prod.state'
    log = f'{path}/prod.csv'

    prod(inpcrd, prmtop, eq_chkpt, traj, rst, chkpt, state, log, n_steps)
