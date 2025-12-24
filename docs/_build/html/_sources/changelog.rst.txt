Changelog
=========

All notable changes to molecular-simulations will be documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_, and this 
project adheres to `Semantic Versioning <https://semver.org/>`_.

v0.3.28 (2025)
--------------

Current release.

Added
~~~~~

* MM-PBSA binding free energy calculations with frame-level parallelization
* ipSAE interface scoring for AlphaFold-predicted structures
* Relative SASA calculations
* Support for constant-pH simulations
* Parsl settings for ALCF Polaris

Changed
~~~~~~~

* Improved test coverage to ~50%
* Updated minimum OpenMM version to 8.0

Fixed
~~~~~

* OpenBLAS threading conflicts in HPC environments
* PDB format handling for non-standard residues

v0.3.0
------

Added
~~~~~

* Protein-protein interaction analysis (PPInteractions)
* Automatic clustering with KMeans++
* Interaction energy fingerprinting
* HPC deployment via Parsl

v0.2.0
------

Added
~~~~~

* OpenMM simulation runner with automatic equilibration
* Explicit and implicit solvent builders
* Small molecule parameterization with GAFF2

v0.1.0
------

Initial release.

* Basic system building with tleap
* Simple simulation runner
