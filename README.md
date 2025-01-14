# molecular-dynamics
Scripts for MD simulation building, running and analysis.

## build.py
Leverages classes in `build/build_amber.py` for building implicit solvent,
explicit solvent and biomolecule + small molecule ligand systems. Can handle
multicomponent systems out of the box so long as the correct force fields are
loaded. Additionally supports the polarizable protein ff amber15ipq although
this remains untested.

## analyze.py
Leverages classes found in `analysis/python/analyzer.py` for performing
analysis using the MDAnalysis library. This was chosen due to its ongoing
development and ease of object-oriented framework as well as straightforward
parallelization.

## simulate.py
Sets up OpenMM simulation objects and performs simple equilibrium simulation
of both implicit and explicit solvent simulations.
