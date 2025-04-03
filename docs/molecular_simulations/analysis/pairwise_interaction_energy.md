Module molecular_simulations.analysis.pairwise_interaction_energy
=================================================================

Classes
-------

`PairwiseInteractionEnergy(topology: str, trajectory: str, sel1: str, sel2: str, stride: int = 1, datafile: str = 'energies.npy')`
:   

    ### Methods

    `build_simulation_object(self, top: openmm.app.topology.Topology, sys: openmm.openmm.System) ‑> openmm.app.simulation.Simulation`
    :

    `calc_energy(self, simulation: openmm.app.simulation.Simulation, positions: numpy.ndarray) ‑> float`
    :

    `compute_energy(self) ‑> None`
    :

    `initialize_systems(self) ‑> None`
    :

    `run(self) ‑> None`
    :

    `split_off_components(self) ‑> None`
    :

    `subset_traj(self, sub_ind: list[str]) ‑> Tuple[openmm.app.topology.Topology, openmm.openmm.System]`
    :

    `unset_charges(self, sel: str) ‑> openmm.app.simulation.Simulation`
    :

    `write_out_energies(self) ‑> None`
    :