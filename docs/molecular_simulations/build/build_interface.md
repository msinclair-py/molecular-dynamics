Module molecular_simulations.build.build_interface
==================================================

Classes
-------

`InterfaceBuilder(path: str | pathlib.Path, pdb: str, interfaces: Dict[str, ...], target: str | pathlib.Path, binder: str | pathlib.Path, padding: float = 10.0, protein: bool = True, rna: bool = False, dna: bool = False, polarizable: bool = False)`
:   Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.
    
    For a given target/binder pair, build systems for driving binding to
    each of the supplied interfaces using DeepDriveMD. Includes writing
    out the required yaml files for running DeepDrive.

    ### Ancestors (in MRO)

    * molecular_simulations.build.build_amber.ExplicitSolvent
    * molecular_simulations.build.build_amber.ImplicitSolvent

    ### Methods

    `build_all(self)`
    :   Iterates through each interface site for a given target and
        builds the corresponding system for the supplied miniprotein
        binder.

    `merge_proteins(self, binder: MDAnalysis.core.groups.AtomGroup) ‑> None`
    :   Merges the target and binder AtomGroups and writes out
        a unified PDB at `self.pdb` so as to leverage the existing
        pipeline for building explicit solvent systems.

    `parse_interface(self, site: str = 'site0') ‑> Dict[str, ...]`
    :   Returns the relevant data for the current interface site.

    `place_binder(self, vector: numpy.ndarray, com: numpy.ndarray) ‑> None`
    :   Move binder nearby to the interface as defined by `vector`. Returns
        an MDAnalysis AtomGroup for the binder.

    `write_cvae_yaml(self, input_shape: List[int]) ‑> None`
    :   Writes the CVAE options yaml for a DeepDriveMD
        simulation.

    `write_ddmd_yaml(self, contact_selection: str, distance_selection: str) ‑> None`
    :   Writes the simulation options yaml for a DeepDriveMD
        simulation.