#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('iondensity.txt',delim_whitespace=True,names=['x','y','ions'])

df = data.groupby(['x','y']).mean()

heatmap_data = pd.pivot_table(df,values='ions',
				index=['x'],
				columns=['y'])

sns.heatmap(heatmap_data)
plt.show()
