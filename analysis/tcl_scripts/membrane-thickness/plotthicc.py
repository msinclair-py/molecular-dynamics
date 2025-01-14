#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ringwidth = 3
data = pd.read_csv('thickness.txt',delim_whitespace=True,usecols=(1,2),names=['ring','thickness'])

data['ring'] = data['ring'] * ringwidth 

sns.lineplot(x='ring',y='thickness',data=data,err_style='bars')
plt.title('Thickness of Membrane Radially From Protein')
plt.xlabel('Distance from Protein (A)')
plt.ylabel('Thickenss (A)')
plt.show()
