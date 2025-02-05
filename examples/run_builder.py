#!/usr/bin/env python
from molecular_simulations.build.build_amber import ExplicitSolvent

path = '.'

for m in range(4):
    mpath = f'{path}/test_sim/model{m}'
    pdb = f'{path}/test_build/chai-model.pdb'

    builder = ExplicitSolvent(mpath, pdb, protein=True)
    builder.build()
