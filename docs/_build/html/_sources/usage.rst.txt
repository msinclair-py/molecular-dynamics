Usage
=====

.. _installation:
.. _examples:

Installation
------------

To use molecular-simulations, first install it using pip:

.. code-block:: shell

   $ pip install molecular-simulations -U

Examples
--------
.. code-block:: python
    :linenos:

    from molecular_simulations.build import ExplicitSolvent
    from molecular_simulations.simulate import Simulator
    from pathlib import Path

    input_pdb = Path('protein.pdb').resolve()
    builder = ExplicitSolvent(input_pdb)
    builder.build()

    sim = Simulator(builder.out, builder.out.with_suffix('.inpcrd'))
    sim.run()
