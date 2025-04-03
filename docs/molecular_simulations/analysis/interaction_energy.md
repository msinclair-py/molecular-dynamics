Module molecular_simulations.analysis.interaction_energy
========================================================

Classes
-------

`DynamicInteractionEnergy(top: str, traj: str, stride: int = 1, chain: str = 'A', platform: str = 'CUDA', first_residue: int | None = None, last_residue: int | None = None, progress_bar: bool = False)`
:   

    ### Methods

    `build_system(self, top: str, traj: str) ‑> openmm.openmm.System`
    :

    `compute_energies(self) ‑> None`
    :

    `load_traj(self, top: str, traj: str) ‑> numpy.ndarray`
    :

    `setup_pbar(self) ‑> None`
    :

`DynamicPotentialEnergy(top: str, traj: str, seltext: str = 'protein')`
:   Class to compute the interaction energy from MD simulation using OpenMM.
    Inspired by: https://github.com/openmm/openmm/issues/3425

    ### Methods

    `build_fake_topology(self, prmtop: bool = True) ‑> None`
    :

    `calc_energy(self, positions) ‑> float`
    :

    `compute(self) ‑> None`
    :

`InteractionEnergy()`
:   Helper class that provides a standard way to create an ABC using
    inheritance.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Methods

    `compute(self)`
    :

    `energy(self)`
    :

    `get_selection(self)`
    :

`InteractionEnergyFrame(system: openmm.openmm.System, top: openmm.app.topology.Topology, chain: str = 'A', platform: str = 'CUDA', first_residue: int | None = None, last_residue: int | None = None)`
:   Computes the linear interaction energy between specified chain and other simulation
    components. Can specify a range of residues in chain to limit calculation to. Works on
    a static model but can be adapted to run on dynamics data.
    
    Inputs:
        pdb (str): Path to input PDB file
        chain (str): Defaults to A. The chain for which to compute the energy between.
            Computes energy between this chain and all other components in PDB file.
        first_residue (int, None): If set, will restrict the calculation to residues
            beginning with resid `first_residue`.
        last_residue (int, None): If set, will restrict the calculation to residues
            ending with resid `last_residue`.

    ### Ancestors (in MRO)

    * molecular_simulations.analysis.interaction_energy.StaticInteractionEnergy

    ### Methods

    `get_system(self)`
    :

`PairwiseInteractionEnergy(topology: str, trajectory: str, sel1_resids: List[int], sel2_resids: List[int], cmat: float | numpy.ndarray, prob_cutoff: float = 0.2, stride: float = 10, platform: str = 'CUDA')`
:   Computes the pairwise interaction energy between a single residue from one 
    selection and the entirety of another selection.

    ### Static methods

    `energy(context, solute_coulomb_scale: int = 0, solute_lj_scale: int = 0, solvent_coulomb_scale: int = 0, solvent_lj_scale: int = 0) ‑> float`
    :

    ### Methods

    `build_context(self, system: openmm.openmm.System, selection: List[int]) ‑> openmm.openmm.Context`
    :

    `compute(self, indices: numpy.ndarray, resid: int) ‑> Dict[str, numpy.ndarray]`
    :   Subsets trajectory based on input indices. Then runs energy analysis per-frame
        on subset trajectory. Returns a dictionary with structure as follows:
            {'lennard-jones': np.ndarray, 'coulombic': np.ndarray}

    `compute_contact_matrix(self, cutoff: float = 10.0) ‑> None`
    :   Computes contact probability matrix for two selections over the course
        of a simulation trajectory. Masks diagonal elements so as not to artificially
        report self-contacts.

    `compute_contacts(self) ‑> None`
    :   Refines full residue selection to only those residues which have a contact
        probability higher than our cutoff (defaults to 0.2).

    `frame_energy(self, context, positions) ‑> Tuple[float]`
    :

    `load(self) ‑> dict`
    :

    `run(self, chkpt_freq=10)`
    :

    `save(self, energies: dict) ‑> None`
    :

    `subset_system(self, sub_ind: List[int])`
    :   Subsets an OpenMM system by a list of atom indices. Should include the residue of
        interest and all other components to measure interaction energy between.

`StaticInteractionEnergy(pdb: str, chain: str = 'A', platform: str = 'CUDA', first_residue: int | None = None, last_residue: int | None = None)`
:   Computes the linear interaction energy between specified chain and other simulation
    components. Can specify a range of residues in chain to limit calculation to. Works on
    a static model but can be adapted to run on dynamics data.
    
    Inputs:
        pdb (str): Path to input PDB file
        chain (str): Defaults to A. The chain for which to compute the energy between.
            Computes energy between this chain and all other components in PDB file.
        first_residue (int, None): If set, will restrict the calculation to residues
            beginning with resid `first_residue`.
        last_residue (int, None): If set, will restrict the calculation to residues
            ending with resid `last_residue`.

    ### Descendants

    * molecular_simulations.analysis.interaction_energy.InteractionEnergyFrame

    ### Static methods

    `energy(context, solute_coulomb_scale: int = 0, solute_lj_scale: int = 0, solvent_coulomb_scale: int = 0, solvent_lj_scale: int = 0) ‑> float`
    :

    ### Methods

    `compute(self, positions: numpy.ndarray | None = None) ‑> None`
    :

    `get_selection(self, topology) ‑> None`
    :

    `get_system(self) ‑> None`
    :