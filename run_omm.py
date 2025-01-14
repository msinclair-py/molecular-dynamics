#!/usr/bin/env python
from glob import glob
from math import ceil
from .simulate.omm_funcs import run_md
import parsl
from .simulate.parsl_config import get_config
import os

# Housekeeping stuff
run_dir = '/eagle/projects/FoundEpidem/msinclair/ideals/whsc1'

paths = sorted(glob('sims/*'))[:-1]
n_jobs = len(paths)
gpus_per_node = 4 # do not change this; 4 per node on Polaris
n_nodes = n_jobs // gpus_per_node # not worth the risk of `ceil`ing a bad flop (e.g. 1.00001)

# PBS options for Parsl
user_opts = {
        'worker_init': f'conda activate simulate; cd {run_dir}',
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
        'worker_init': f'module use /soft/modulefiles/; module load conda; conda activate simulate; cd {run_dir}; export PYTHONPATH=$PYTHONPATH:{run_dir}',
        'scheduler_options': '#PBS -l filesystems=home:eagle',
        'account': 'FoundEpidem',
        'queue': 'debug',
        'walltime': '0:30:00',
        'nodes_per_block': n_nodes,
        'cpus_per_node': 32,
        'available_accelerators': 4,
        'cores_per_worker': 8
        }

cfg = get_config(run_dir, debug_opts)

sim_length = 500 # ns
timestep = 4 # fs
n_steps = sim_length / timestep * 1000000

run_md = parsl.python_app(run_md)

parsl_cfg = parsl.load(cfg)

futures = []
for path in paths:
    futures.append(run_md(path, n_steps=n_steps))

outputs = [x.result() for x in futures]
