Basic Protein Simulation
========================

This tutorial walks through running a complete molecular dynamics simulation of a 
protein, from structure preparation to trajectory analysis.

Prerequisites
-------------

* A protein structure in PDB format
* AmberTools installed and accessible
* OpenMM with GPU support (recommended)

We'll use lysozyme (PDB: 1AKI) as our example system.

Step 1: Build the System
------------------------

Create an explicitly solvated system with the ff19SB force field:

.. code-block:: python

   from molecular_simulations.build import ExplicitSolvent

   pdb_file = Path('/path/to/1AKI.pdb')
   output_dir = Path("./lysozyme_sim")

   builder = ExplicitSolvent(
       out=output_dir,
       pdb=pdb_file,
       # Optional parameters:
       # padding=12.0,     # Angstroms of water padding
       # amberhome='/path/to/env',  # Use another env's AmberTools installation
   )
   builder.build()

This creates:

* ``system.prmtop`` - AMBER topology with force field parameters
* ``system.inpcrd`` - Initial coordinates with solvent and ions

.. note::
   
   The builder uses tleap internally. Check the generated ``tleap.log`` file 
   if you encounter issues with unusual residues or missing parameters.

Step 2: Run the Simulation
--------------------------

Initialize and run the simulation with default settings:

.. code-block:: python

   from molecular_simulations.simulate import Simulator

   sim = Simulator(
       path=output_dir,
       equil_steps=500_000,     # 1 ns equilibration
       prod_steps=5_000_000,    # 20 ns production
   )
   sim.run()

The simulator automatically:

1. Minimizes the system (steepest descent)
2. Heats from 0 K to 300 K with position restraints
3. Equilibrates in NPT ensemble
4. Runs production dynamics with 4 fs timestep (hydrogen mass repartitioning)

Output files:

* ``prod.dcd`` - Production trajectory
* ``prod.log`` - Energy, temperature, and density data

Customizing Simulation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control over the simulation:

.. code-block:: python

   sim = Simulator(
       path=output_dir,
       equil_steps=2_500_000,         # 5 ns equilibration total (unrestrained + restrained)
       prod_steps=25_000_000,         # 100 ns production @ 4 fs timestep
       temperature=310,               # K (body temperature)
       n_equil_cycles=3,              # N cycles of unrestrained equilibration
       prod_reporter_frequency=5000,  # Save every 20 ps @ 4 fs timestep
       platform="CUDA",               # Force GPU platform
   )

Step 3: Basic Analysis
----------------------

Check the trajectory looks reasonable with basic metrics:

.. code-block:: python

   import MDAnalysis as mda
   from MDAnalysis.analysis import rms

   # Load the trajectory
   u = mda.Universe(
       str(output_dir / "system.prmtop"),
       str(output_dir / "prod.dcd")
   )

   # Calculate RMSD relative to first frame
   protein = u.select_atoms("protein and name CA")
   R = rms.RMSD(protein, ref_frame=0).run()

   print(f"Trajectory: {len(u.trajectory)} frames")
   print(f"Final RMSD: {R.results.rmsd[-1, 2]:.2f} Å")

Step 4: Interaction Fingerprinting
----------------------------------

Calculate per-residue interaction energies to identify key contacts:

.. code-block:: python

   from molecular_simulations.analysis import Fingerprinter

   fp = Fingerprinter(
       topology=str(output_dir / "system.prmtop"),
       trajectory=str(output_dir / "prod.dcd"),
       target_selection="protein",  # Entire protein
   )
   fp.run()
   fp.save()  # Creates fingerprint.npz

   # Load and examine results
   import numpy as np
   data = np.load("fingerprint.npz")
   print(f"Residues analyzed: {data['residues'].shape[0]}")

Step 5: Clustering Conformations
--------------------------------

Identify distinct conformational states:

.. code-block:: python

   from molecular_simulations.analysis import AutoKMeans

   clusterer = AutoKMeans(
       data_directory=str(output_dir),  # Directory with .npy feature files
       max_clusters=5,
       reduction_algorithm="PCA",
       reduction_kws={"n_components": 2},
   )
   clusterer.run()
   clusterer.save_labels()
   clusterer.save_centers()

   print(f"Optimal clusters: {clusterer.n_clusters_}")

Troubleshooting
---------------

**Simulation crashes immediately**
   Check that OpenMM can access the GPU: ``python -c "import openmm; print(openmm.Platform.getNumPlatforms())"``

**System has bad contacts after building**
   The initial minimization should resolve this. If it persists, check the input 
   PDB for unusual conformations or clashing atoms.

**Trajectory file is empty or truncated**
   Check ``prod.log`` for errors. Common issues include disk space and unstable 
   simulations (check temperature/energy for spikes).

Next Steps
----------

* :doc:`ligand_systems` - Add small molecules to your simulations
* :doc:`hpc_deployment` - Run longer simulations on clusters
* :doc:`analysis_workflows` - More sophisticated analysis pipelines
