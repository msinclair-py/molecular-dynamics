#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None # default='warn'

def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	c = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
	return c

numions = 664 

dframe = pd.read_csv('ionpositions.txt',delim_whitespace=True,names=['frame','ion','slab'])

# make a dataframe for each ion; check if it goes from 0->2 or 2->0
for i in range(numions):
	df = dframe.iloc[i::numions]
	df.reset_index(drop=True,inplace=True)

	# get initial position and assign final position
	# NOTE: if ion starts in protein we need to handle it carefully
	init = df.iloc[0,2]
	if init == 0:
		final = 2
	elif init == 2:
		final = 0
	else:
		final = 2

	# check if the ion ever enters the protein
	pattern = np.asarray([1,final])
	N = 2
	arr = df['slab'].values
	m = (rolling_window(arr,N) == pattern).all(1)
	m_ext = np.r_[m,np.zeros(len(arr) - len(m), dtype=bool)]
	df['exit'] = binary_dilation(m_ext, structure=[1]*N, origin=-(N//2))

	if df['exit'].any():
		print(df)
