from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints

# Adjust your user-specific options here:
run_dir="/eagle/projects/FoundEpidem/msinclair/ideals/whsc1"

user_opts_prod = {
    "worker_init":      f"conda activate simulate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle" , # specify any PBS options here, like filesystems
    "account":          "FoundEpidem",
    "queue":            "preemptable",
    "walltime":         "24:00:00",
    "nodes_per_block":  1, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
    "cpus_per_node":    32, # Up to 64 with multithreading
    "available_accelerators": 4, # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.  
}

user_opts_debug = {
    "worker_init":      f"conda activate simulate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle" , # specify any PBS options here, like filesystems
    "account":          "FoundEpidem",
    "queue":            "debug",
    "walltime":         "1:00:00",
    "nodes_per_block":  1, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
    "cpus_per_node":    32, # Up to 64 with multithreading
    "available_accelerators": 4, # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.  
}

user_opts = user_opts_prod

checkpoints = get_all_checkpoints(run_dir)
print("Found the following checkpoints: ", checkpoints)

config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=15,
                heartbeat_threshold=120,
                worker_debug=True,
                available_accelerators=user_opts["available_accelerators"], # if this is set, it will override other settings for max_workers if set
                cores_per_worker=user_opts["cores_per_worker"],
                address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                # start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    # Which launcher to use?  Check out the note below for some details.  Try MPI first!
                    # launcher=GnuParallelLauncher(),
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    select_options="ngpus=4",
                    # PBS directives (header lines): for array jobs pass '-J' option
                    scheduler_options=user_opts["scheduler_options"],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts["worker_init"],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts["nodes_per_block"],
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1, # Can increase more to have more parallel jobs
                    cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"]
                ),
            ),
        ],
        checkpoint_files = checkpoints,
        run_dir=run_dir,
        checkpoint_mode = 'task_exit',
        retries=2,
        app_cache=True,
)
