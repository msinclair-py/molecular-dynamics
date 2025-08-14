Usage
=====

Installation
------------
.. code-block:: console

   $ pip install molecular-simulations -U

Example
-------
.. code-block:: python

   from molecular_simulations.build import ExplicitSolvent
   from molecular_simulations.simulate import Simulator
   from pathlib import Path

   pdb_file = Path('/path/to/file.pdb')
   out_path = Path('/path/to/sim_inputs')
   builder = ExplicitSolvent(out_path, pdb_file)
   builder.build()

   sim = Simulator(builder.out, builder.out.with_suffix('inpcrd'))
   sim.run()
