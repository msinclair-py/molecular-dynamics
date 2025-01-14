#!/usr/bin/env python
import MDAnalysis as md
import numpy as np
import pandas as pd

u = md.Universe(orco.psf,md1.dcd)
waters = u.select_atoms('segname TIP3 and within 5 of ((protein and resid xx to yy) or (protein and resid xx to yy))',updating=True)

raw_data = np.zeros((len(u.trajectory),3))

for ts in u.trajectory:
	for water in waters:
		res = water.getresid
		raw_data[ts] = [ts,res,water.position[2]]


set fo [open "surfwaters.txt" w]
