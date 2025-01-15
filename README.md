# molecular-dynamics
Scripts for MD simulation building, running and analysis.

## run_builder.py
Leverages classes in `build/build_amber.py` for building implicit solvent,
explicit solvent and biomolecule + small molecule ligand systems. Can handle
multicomponent systems out of the box so long as the correct force fields are
loaded. Additionally supports the polarizable protein ff amber15ipq although
this remains untested.

## run_sim_analysis.py
Leverages classes found in `analysis/analyzer.py` for performing
analysis using the MDAnalysis library. This was chosen due to its ongoing
development and ease of object-oriented framework as well as straightforward
parallelization.

## run_omm.py
Sets up OpenMM simulation objects and performs simple equilibrium simulation
of both implicit and explicit solvent simulations using Parsl. Configured
by default to run on the Argonne ALCF Polaris supercomputer but can be 
adapted for any scheduler on any cluster by modifying the Parsl configuration.
