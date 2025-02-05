#!/usr/bin/env python
from glob import glob
from math import ceil
import parsl
from molecular_simulations.simulate import parsl_config
import os

# Housekeeping stuff
run_dir = '/eagle/projects/FoundEpidem/msinclair/tools/molecular-dynamics/examples'

paths = sorted([f for f in glob('test_sims/*')])
n_jobs = len(paths)
gpus_per_node = 4 # do not change this; 4 per node on Polaris
n_nodes = n_jobs // gpus_per_node # not worth the risk of `ceil`ing a bad flop (e.g. 1.00001)

worker_init_cmd = f'module use /soft/modulefiles/; module load conda; conda activate simulate; \
        cd {run_dir}'

# PBS options for Parsl
user_opts = {
        'worker_init': worker_init_cmd,
        'scheduler_options': '#PBS -l filesystems=home:eagle',
        'account': 'FoundEpidem',
        'queue': 'preemptable',
        'walltime': '72:00:00',
        'nodes_per_block': n_nodes,
        'cpus_per_node': 32,
        'available_accelerators': 4,
        'cores_per_worker': 8
        }

debug_opts = {
        'worker_init': worker_init_cmd,
        'scheduler_options': '#PBS -l filesystems=home:eagle',
        'account': 'FoundEpidem',
        'queue': 'debug',
        'walltime': '0:30:00',
        'nodes_per_block': n_nodes,
        'cpus_per_node': 32,
        'available_accelerators': 4,
        'cores_per_worker': 8
        }

cfg = parsl_config.get_config(run_dir, user_opts)

sim_length = 10 # ns
timestep = 4 # fs
n_steps = sim_length / timestep * 1000000

@parsl.python_app
def run_md(path: str, eq_steps=500_000, steps=250_000_000):
    from molecular_simulations.simulate.omm_simulator import Simulator

    simulator = Simulator(path, equil_steps=eq_steps, prod_steps=steps)
    simulator.run()

parsl_cfg = parsl.load(cfg)

futures = []
for path in paths:
    futures.append(run_md(path, steps=n_steps))

outputs = [x.result() for x in futures]
