#!/usr/bin/env python
from glob import glob
import MDAnalysis as mda
from natsort import natsorted
import os
from .analysis.analyzer import SimulationAnalyzer
from tqdm import tqdm

path = '/eagle/projects/FoundEpidem/avasan/IDEAL/MiniBinderSims/complexes/sims/tasks'
sim_list = natsorted(list(glob(f'{path}/*')))

for syst in tqdm(sim_list):
    p, s = os.path.split(syst)
    A = SimulationAnalyzer(p, s)
    if isinstance(A.u, mda.Universe) and A.length > 1:
        A.analyze()
