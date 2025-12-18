HPC Deployment with Parsl
=========================

This tutorial covers deploying simulations on high-performance computing clusters 
using Parsl for workflow management.

Overview
--------

molecular-simulations integrates with `Parsl <https://parsl.readthedocs.io/>`_ to 
enable:

* Running multiple simulation replicas in parallel
* Automatic job submission to PBS/SLURM schedulers
* GPU allocation across nodes
* Fault tolerance and checkpointing

Configuration Files
-------------------

Create a YAML configuration file for your cluster:

.. code-block:: yaml
   :caption: parsl_config.yaml

   # Local workstation with multiple GPUs
   executor: ThreadPoolExecutor
   max_workers: 4

   # PBS cluster example
   # executor: HighThroughputExecutor
   # provider: PBSProProvider
   # account: "myproject"
   # queue: "prod"
   # walltime: "24:00:00"
   # nodes_per_block: 1
   # available_accelerators: 4

Using LocalSettings
-------------------

For local workstations or simple clusters:

.. code-block:: python

   import parsl
   from molecular_simulations.simulate import LocalSettings
   from pathlib import Path

   # Load configuration
   settings = LocalSettings.from_yaml("parsl_config.yaml")
   config = settings.config_factory("/path/to/run_dir")
   parsl.load(config)

   # Define the simulation app
   @parsl.python_app
   def run_md(path: str, steps: int = 25_000_000):
       from molecular_simulations.simulate import Simulator
       Simulator(path, prod_steps=steps).run()
       return path

   # Submit jobs for all replicas
   replica_dirs = list(Path("./").glob("replica_*"))
   futures = [run_md(str(p)) for p in replica_dirs]

   # Wait for completion
   results = [f.result() for f in futures]
   print(f"Completed {len(results)} simulations")

Using PolarisSettings
---------------------

For ALCF Polaris supercomputer:

.. code-block:: python

   from molecular_simulations.simulate import PolarisSettings

   settings = PolarisSettings(
       account="myproject",
       queue="prod",
       walltime="12:00:00",
       nodes_per_block=10,
       worker_init="module load cudatoolkit; source activate molsim",
   )
   config = settings.config_factory("/path/to/run_dir")

Best Practices
--------------

**Organize replica directories**
   Use a consistent naming scheme like ``replica_001/``, ``replica_002/``, etc.

**Set appropriate walltime**
   Estimate based on system size and simulation length. Add buffer for 
   equilibration and I/O. Ensure you do not ask for more than allowed
   on your resource as this will crash Parsl with no discernable error.

**Use checkpointing**
   For long simulations, configure periodic checkpoint saving to enable 
   restart from failures.

Troubleshooting
---------------

**Jobs fail immediately**
   Check that the worker_init script correctly loads all required modules 
   and activates the conda/virtual environment. Ensure paths are either
   correct if relative or absolute. Remember that the `run_dir` arg for the
   config factory is where Parsl will sit in the job, meaning paths are relative
   to this location.

**OpenBLAS threading errors**
   Set ``OMP_NUM_THREADS=1`` in worker_init to avoid conflicts with MMPBSA's 
   internal threading.

**Errors do not propagate into runtime logs**
   Check in the Parsl output logs. Sometimes errors appear in the `parsl.log`,
   but more often you will need to look at the error stream located at
   `00*/submit_scripts/parsl.*.sh.err`. Most python tracebacks end up here.
   
**I have tried everything and now I hate Parsl**
   Yes, Parsl can be challenging to debug at times. My best advice if all else
   fails is to try running the code in serial to see what bugs arise. Nearly 
   every time I am convinced Parsl is to blame, it is actually just a bug in
   the code, an unstable system, etc.
