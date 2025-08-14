#!/usr/bin/env python
from molecular_simulations.simulate import Simulator
from pathlib import Path

path = Path('/path/to/simulation/inputs')
sim_length = 10 # ns
timestep = 4 # fs

n_steps = sim_length / timestep * 1000000 # production steps
eq_steps = 500_000 # 1ns; 2 fs timestep

simulator = Simulator(path, equil_steps=eq_steps, prod_steps=steps)
simulator.run()
