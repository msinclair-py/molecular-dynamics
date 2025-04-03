Module molecular_simulations.build.build_amber
==============================================

Classes
-------

`ComplexBuilder(path: str, pdb: str, lig: str, padding: float = 10.0, **kwargs)`
:   Builds complexes consisting of a biomolecule pdb and small molecule ligand.
    Runs antechamber workflow to generate gaff2 parameters.

    ### Ancestors (in MRO)

    * molecular_simulations.build.build_amber.ExplicitSolvent
    * molecular_simulations.build.build_amber.ImplicitSolvent

    ### Methods

    `add_hydrogens(self) ‑> None`
    :   Add hydrogens in rdkit. Atom hybridization is taken from the
        input sdf file and if this is incorrect, hydrogens will be wrong
        too.

    `assemble_system(self, dim, num_ions) ‑> None`
    :   Slightly modified from the parent class, now we have to add
        the ligand parameters and assemble a complex rather than just
        placing a biomolecule in the water box.

    `check_sqm(self) ‑> None`
    :   Checks for evidence that antechamber calculations exited
        successfully. This is always on the second to last line,
        and if not present, indicates that we failed to produce
        sane parameters for this molecule. In that case, I wish
        you good luck.

    `move_antechamber_outputs(self) ‑> None`
    :   Remove unneccessary outputs from antechamber. Keep the
        sqm.out file as proof that antechamber did not fail.

    `parameterize_ligand(self)`
    :   Ensures consistent treatment of all ligand sdf files, generating
        GAFF2 parameters in the form of .frcmod and .lib files. Produces
        a mol2 file for coordinates and connectivity and ensures that
        antechamber did not fail. Hydrogens are added in rdkit which 
        generally does a good job of this.

`ExplicitSolvent(path: str, pdb: str, padding: float = 10.0, protein: bool = True, rna: bool = False, dna: bool = False, polarizable: bool = False)`
:   Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.

    ### Ancestors (in MRO)

    * molecular_simulations.build.build_amber.ImplicitSolvent

    ### Descendants

    * molecular_simulations.build.build_amber.ComplexBuilder
    * molecular_simulations.build.build_interface.InterfaceBuilder

    ### Static methods

    `get_ion_numbers(volume: int) ‑> float`
    :   Returns the number of Chloride? ions required to achieve 150mM
        concentration for a given volume. The number of Sodium counter
        ions should be equivalent.

    ### Methods

    `assemble_system(self, dim: float, num_ions: int) ‑> None`
    :   Build system in tleap.

    `clean_up_directory(self) ‑> None`
    :   Remove leap log. This is placed wherever the script calling it
        runs and likely will throw errors if multiple systems are
        being iteratively built.

    `get_pdb_extent(self) ‑> int`
    :   Identifies the longest axis of the protein in terms of X/Y/Z
        projection. Not super accurate but likely good enough for determining
        PBC box size. Returns longest axis length + 2 times the padding
        to account for +/- padding.

    `prep_pdb(self)`
    :

    `write_leap(self, inp: str) ‑> str`
    :   Writes out a tleap input file and returns the path
        to the file.

`ImplicitSolvent(path: str | pathlib.Path, pdb: str, protein: bool = True, rna: bool = False, dna: bool = False, phos_protein: bool = False, mod_protein: bool = False, out: str | pathlib.Path | None = None)`
:   Class for building a system using ambertools. Produces explicit solvent cubic box
    with user-specified padding which has been neutralized and ionized with 150mM NaCl.

    ### Descendants

    * molecular_simulations.build.build_amber.ExplicitSolvent
    * molecular_simulations.build.build_amber.PLINDERBuilder

    ### Methods

    `build(self) ‑> None`
    :   Orchestrate the various things that need to happen in order
        to produce an explicit solvent system. This includes running
        `pdb4amber`, computing the periodic box size, number of ions
        needed and running `tleap` to make the final system.

    `pdbfixit(self, add_hydrogen=True) ‑> str`
    :

`LigandBuilder(path: str | pathlib.Path, lig: str, lig_number: int = 0)`
:   

    ### Methods

    `add_hydrogens(self) ‑> None`
    :   Add hydrogens in rdkit. Atom hybridization is taken from the
        input sdf file and if this is incorrect, hydrogens will be wrong
        too.

    `check_sqm(self) ‑> int`
    :   Checks for evidence that antechamber calculations exited
        successfully. This is always on the second to last line,
        and if not present, indicates that we failed to produce
        sane parameters for this molecule. In that case, I wish
        you good luck.

    `convert_to_mol2(self) ‑> None`
    :

    `move_antechamber_outputs(self) ‑> None`
    :   Remove unneccessary outputs from antechamber. Keep the
        sqm.out file as proof that antechamber did not fail.

    `parameterize_ligand(self) ‑> None`
    :   Ensures consistent treatment of all ligand sdf files, generating
        GAFF2 parameters in the form of .frcmod and .lib files. Produces
        a mol2 file for coordinates and connectivity and ensures that
        antechamber did not fail. Hydrogens are added in rdkit which 
        generally does a good job of this.

    `write_leap(self, inp: str) ‑> str`
    :   Writes out a tleap input file and returns the path
        to the file.

`LigandError(message='This system contains ligands which we cannot model!')`
:   Common base class for all non-exit exceptions.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

`PLINDERBuilder(path: str | pathlib.Path, system_id: str, out: str | pathlib.Path, **kwargs)`
:   Builds complexes consisting of a biomolecule pdb and small molecule ligand.
    Runs antechamber workflow to generate gaff2 parameters.

    ### Ancestors (in MRO)

    * molecular_simulations.build.build_amber.ImplicitSolvent

    ### Instance variables

    `anion_list: List[str]`
    :

    `cation_list: List[str]`
    :

    ### Methods

    `assemble_system(self) ‑> None`
    :   Slightly modified from the parent class, now we have to add
        the ligand parameters and assemble a complex rather than just
        placing a biomolecule in the water box.

    `check_ligand(self, ligand: str | pathlib.Path) ‑> bool`
    :   Check ligand for ions and other weird stuff. We need to take care not
        to assume all species containing formal charges are ions, nor that all
        species containing atoms in the cation/anion lists are ions. Good example
        is the multitude of small molecule drugs containing bonded halogens.

    `check_ptms(self, sequence: List[str], chain_residues: List[str]) ‑> List[str]`
    :   Check the full sequence (from fasta) against the potentially partial
        sequence from the structural model stored in `chain_residues`.

    `inject_fasta(self, chain_map: Dict[str, List[str]]) ‑> List[pdbfixer.pdbfixer.Sequence]`
    :   Checks fasta against actual sequence. Modifies sequence so that 
        it correctly matches in the case of non-canonical residues such
        as phosphorylations (i.e. SER -> SEP).

    `ligand_handler(self, ligs: List[str | pathlib.Path]) ‑> List[str | pathlib.Path]`
    :

    `migrate_files(self) ‑> List[str]`
    :

    `place_ions(self) ‑> None`
    :   This is horrible and I apologize profusely if you find yourself
        having to go through the following. Good luck.

    `prep_protein(self) ‑> None`
    :

    `triage_pdb(self, broken_pdb: str | pathlib.Path, repaired_pdb: str | pathlib.Path) ‑> str`
    :   Runs PDBFixer to repair missing loops and ensure structure is
        in good shape. Runs a check against the sequence provided by
        PLINDER and ensures that any non-canonical residues are represented
        in the sequence properly.