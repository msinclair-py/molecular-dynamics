#!/usr/bin/env python
from abc import ABC, abstractmethod
import json
from parsl.config import Config
from parsl.providers import LocalProvider, PBSProProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
from parsl.addresses import address_by_interface, address_by_hostname
from parsl.utils import get_all_checkpoints
from pathlib import Path
from pydantic import BaseModel
from typing import List, Sequence, Tuple, Type, TypeVar, Union
import yaml

PathLike = Union[str, Path]
_T = TypeVar("_T")

class BaseSettings(BaseModel):
    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore

class BaseComputeSettings(ABC, BaseSettings):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    @abstractmethod
    def config_factory(self, run_dir: PathLike) -> Config:
        """
        Create new Parsl configuration.
        """

class LocalSettings(BaseComputeSettings):
    available_accelerators: Union[int, Sequence[str]] = 4
    retries: int = 1
    label: str = 'htex'
    worker_port_range: Tuple[int, int] = (10000, 20000)

    def config_factory(self) -> Config:
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address="127.0.0.1",
                    label=self.label,
                    cpu_affinity="block",
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1)
                ),
            ],
        )

class PolarisSettings(BaseComputeSettings):
    label: str = 'htex'
    num_nodes: int = 1
    worker_init: str = ''
    scheduler_options: str = ''
    account: str
    queue: str
    walltime: str
    cpus_per_node: int = 64
    strategy: str = 'simple'
    available_accelerators: Union[int, Sequence[str]] = 4

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a configuration suitable for running all tasks on single nodes of Polaris
        We will launch 4 workers per node, each pinned to a different GPU
        Args:
            num_nodes: Number of nodes to use for the MPI parallel tasks
            user_options: Options for which account to use, location of environment files, etc
            run_dir: Directory in which to store Parsl run files. Default: `runinfo`
        """
        return Config(
            retries=1,  # Allows restarts if jobs are killed by the end of a job
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    available_accelerators=self.available_accelerators,  
                    address=address_by_hostname(),
                    cpu_affinity="alternating",
                    prefetch_capacity=0,  # Increase if you have many more tasks than workers
                    provider=PBSProProvider(  # type: ignore[no-untyped-call]
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                        ),  # Updates to the mpiexec command
                        account=self.account,
                        queue=self.queue,
                        select_options="ngpus=4",
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options=self.scheduler_options,
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            strategy=self.strategy,
            app_cache=True,
        )

class AuroraSettings(BaseComputeSettings):
    label: str = 'htex'
    worker_init: str = ""
    num_nodes: int = 1
    scheduler_options: str = ""
    account: str
    queue: str
    walltime: str
    retries: int = 0
    cpus_per_node: int = 48 # only 4 cpus per OpenMM job
    strategy: str = "simple"
    available_accelerators: List[str] = [str(i) for i in range(12)]

    def config_factory(self, run_dir: PathLike) -> Config:
        """Create a Parsl configuration for running on Aurora."""
        return Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    available_accelerators=self.available_accelerators,
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers=12,
                    cores_per_worker=16,
                    heartbeat_period=30,
                    heartbeat_threshold=300,
                    worker_debug=False,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind",
                            overrides=f"--depth=208 --ppn 1"
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        account=self.account,
                        queue=self.queue,
                        walltime=self.walltime,

                    ),
                ),
            ],
            run_dir=str(run_dir),
            checkpoint_mode='task_exit',
            retries=self.retries,
            app_cache=True,
        )

