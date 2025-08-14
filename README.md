# molecular-dynamics repository README
[molecular-dynamics](https://molecular-simulations.readthedocs.io/en/latest/index.html) 
is a collection of python utilities for MD simulation building, 
running and analysis.

Capable of building molecular dynamics systems using the AMBER forcefield
ecosystem. Available forcefields include: fixed-charge forcefields (ff19SB 
for proteins, OL21 for DNA, and OL3 for RNA), ff15ipq polarizable forcefield,
and the ability to parameterize small molecules using GAFF2. In all explicit
systems OPC water is used with the exception of polarizable systems where a
polarizable water model is used (SPC/Eb). 
Loosely tested support for implicit solvent systems as well.

Also bundled in this library are tools for analyzing molecular simulations
including: correlation analysis, interaction energy, residue energy footprinting, 
automatic clustering, an MDAnalysis-based SASA and RSASA methods and more.

## Installation
Easily installed via pip. PyPI page: https://pypi.org/project/molecular-simulations/

`pip install molecular-simulations`

## Package details
Simulations are performed using OpenMM version >= 8 due to CUDA versioning
on ALCF Polaris. Note that due to some scaling issues reported by users,
OpenMM versions 8.0-8.1 suffer from slow integration times for larger
systems. All MD simulations are deployed on a PBS scheduler via 
[Parsl](https://parsl.readthedocs.io/en/stable/) but
with some minor tweaking of the Parsl configuration object they can be
deployed on theoretically any HPC cluster/job scheduler. Jobs can also
be run locally without the need for Parsl (example coming soon).

## examples/ contains sample execution scripts
### examples/run_builder.py
Leverages classes in `build/build_amber.py` for building implicit solvent,
explicit solvent and biomolecule + small molecule ligand systems. Can handle
multicomponent systems out of the box so long as the correct force fields are
loaded. Additionally supports the polarizable protein ff amber15ipq although
this remains untested.

### examples/run_sim_analysis.py
Leverages classes found in `analysis/analyzer.py` for performing
analysis using the MDAnalysis library. This was chosen due to its ongoing
development and ease of object-oriented framework as well as straightforward
parallelization.

### examples/run_omm.py
Sets up OpenMM simulation objects and performs simple equilibrium simulation
of both implicit and explicit solvent simulations using Parsl. Configured
by default to run on the Argonne ALCF Polaris supercomputer but can be 
adapted for any scheduler on any cluster by modifying the Parsl configuration.

See Parsl documentation for further examples of how to run on other HPC resources. https://parsl.readthedocs.io/en/stable/
