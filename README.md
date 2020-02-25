# molecular-dynamics
Scripts for MD Analysis


Area Compressibility

This script has two parts, a tcl script for VMD, and a python script
for data analysis. There is a single executable that manages the I/O.

Membrane Thickness

This script evaluates a membrane simulation using a tcl script for VMD.
The resulting data is then analyzed in python and output as a series of
average membrane thickness, for the net membrane average and then the
average of a series of rings surrounded protein of interest of user-defined
length.

Local Hydration

This script analyzes water packing along a membrane surface to detect any
abnormalities that may arise due to the inclusion of a membrane bound 
element.

Ion Conductance

This script bins ion z coordinates into 3 slabs and then detects any ions
that pass from the first to last slab via the middle slab or vice versa.
This ensures that periodicity of the simulation box is accounted for.

Water Profile

This script measures the density of water oxygens along the z coordinate of
a membrane protein's pore. It outputs a relative abundance histogram of the
waters.
