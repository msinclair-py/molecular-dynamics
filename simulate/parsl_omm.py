#!/usr/bin/env python
from glob import glob
from math import ceil
from .omm_funcs import run_md
from .parsl_funcs import get_config

# Housekeeping stuff
run_dir = '/eagle/projects/FoundEpidem/msinclair/ideals/whsc1'

paths = glob('sims/*')
n_jobs = len(paths)
gpus_per_node = 4 # do not change this; 4 per node on Polaris
n_nodes = n_jobs // gpus_per_node # not worth the risk of `ceil`ing a bad flop (e.g. 1.00001)

# PBS options for Parsl
user_opts = {
        'worker_init': f'conda activate simulate; cd {run_dir}'
        'scheduler_options': '#PBS -l filesystems=home:eagle',
        'account': 'FoundEpidem',
        'queue': 'preemptable',
        'walltime': '24:00:00',
        'nodes_per_block': n_nodes,
        'cpus_per_node': 32,
        'available_accelerators': 4,
        'cores_per_worker': 8
        }

cfg = get_config(run_dir, user_opts)

sim_length = 500 # ns
timestep = 4 # fs
n_steps = sim_length / timestep * 1000000

# Loading config spins up parallelism; do the magic here
with parsl.load(cfg):
    for path in paths:
        run_md(path, n_steps=n_steps)
