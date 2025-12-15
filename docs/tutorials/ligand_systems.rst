Protein-Ligand Systems
======================

This tutorial covers working with systems containing small molecule ligands.

.. note::
   
   This tutorial requires the ``ligand`` optional dependencies:
   
   .. code-block:: console
   
      $ pip install molecular-simulations[ligand]

Prerequisites
-------------

* RDKit for molecule handling
* OpenBabel for format conversion
* A protein structure and ligand (SDF/MOL2 format)

Parameterizing Small Molecules
------------------------------

The :class:`~molecular_simulations.build.LigandBuilder` class handles GAFF2 
parameterization:

.. code-block:: python

   from molecular_simulations.build import LigandBuilder
   from pathlib import Path

   ligand_file = Path("ligand.sdf")
   output_dir = Path("./ligand_params")

   builder = LigandBuilder(
       ligand_file=ligand_file,
       output_dir=output_dir,
       charge_method="am1bcc",  # AM1-BCC charges
   )
   builder.build()

   # Outputs:
   # - ligand.mol2 (with charges)
   # - ligand.frcmod (GAFF2 parameters)

Building the Complex
--------------------

Combine the parameterized ligand with your protein:

.. code-block:: python

   from molecular_simulations.build import ExplicitSolvent

   builder = ExplicitSolvent(
       out=Path("./complex_sim"),
       pdb=Path("protein.pdb"),
       ligand_params=output_dir / "ligand.frcmod",
       ligand_mol2=output_dir / "ligand.mol2",
   )
   builder.build()

Running and Analysis
--------------------

Simulation and analysis proceed as with protein-only systems. For interaction 
analysis, specify the ligand selection:

.. code-block:: python

   from molecular_simulations.analysis import Fingerprinter

   fp = Fingerprinter(
       topology="system.prmtop",
       trajectory="prod.dcd",
       target_selection="protein",
       binder_selection="resname LIG",  # Ligand residue name
   )
   fp.run()

Common Issues
-------------

**Ligand parameterization fails**
   Check that the ligand has correct protonation state and no unusual 
   functional groups. GAFF2 may not cover all chemistries.

**Charges don't sum to integer**
   Ensure the input structure has correct formal charges. Consider using 
   ``charge_method="resp"`` for more accurate charges.
