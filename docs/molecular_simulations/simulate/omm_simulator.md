Module molecular_simulations.simulate.omm_simulator
===================================================

Classes
-------

`ImplicitSimulator(path: str, equil_steps: int = 1250000, prod_steps: int = 250000000, n_equil_cycles: int = 3, reporter_frequency: int = 1000, platform: str = 'CUDA', device_ids: list[int] = [0], force_constant: float = 10.0, implicit_solvent: openmm.app.internal.singleton.Singleton = GBn2, solute_dielectric: float = 1.0, solvent_dielectric: float = 78.5)`
:   

    ### Ancestors (in MRO)

    * molecular_simulations.simulate.omm_simulator.Simulator

    ### Methods

    `equilibrate(self) ‑> openmm.app.simulation.Simulation`
    :

    `load_amber_files(self) ‑> openmm.openmm.System`
    :

`Simulator(path: str, equil_steps: int = 1250000, prod_steps: int = 250000000, n_equil_cycles: int = 3, reporter_frequency: int = 1000, platform: str = 'CUDA', device_ids: list[int] = [0], force_constant: float = 10.0)`
:   

    ### Descendants

    * molecular_simulations.simulate.omm_simulator.ImplicitSimulator

    ### Static methods

    `add_backbone_posres(system: openmm.openmm.System, positions: numpy.ndarray, atoms: List[str], indices: List[int], restraint_force: float = 10.0) ‑> openmm.openmm.System`
    :

    ### Methods

    `attach_reporters(self, simulation: openmm.app.simulation.Simulation, dcd_file: str, log_file: str, rst_file: str, restart: bool = False) ‑> openmm.app.simulation.Simulation`
    :

    `equilibrate(self) ‑> openmm.app.simulation.Simulation`
    :

    `get_restraint_indices(self, addtl_selection: str = '') ‑> List[int]`
    :

    `load_amber_files(self) ‑> openmm.openmm.System`
    :

    `load_checkpoint(self, simulation: openmm.app.simulation.Simulation, checkpoint: str) ‑> openmm.app.simulation.Simulation`
    :

    `production(self, chkpt: str, restart: bool = False) ‑> None`
    :

    `run(self) ‑> None`
    :

    `setup_sim(self, system: openmm.openmm.System, dt: float) ‑> Tuple[openmm.app.simulation.Simulation, openmm.openmm.Integrator]`
    :