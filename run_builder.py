#!/usr/bin/env python
from .build.build_amber import ExplicitSolvent

path = '/eagle/projects/FoundEpidem/msinclair/ideals/whsc1'

for m in range(5):
    mpath = f'{path}/sims/model{m}'
    pdb = f'{path}/fold_prot_dna_whsc1/pred.model_idx_{m}.pdb'

    builder = ExplicitSolvent(mpath, pdb, protein=True, dna=True)
    builder.build()
