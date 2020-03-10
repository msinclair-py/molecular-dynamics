#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('surfwaters.txt',delim_whitespace=True,names=['frame','resid','z'])

sns.distplot(data['z'],vertical=True)
plt.xlabel('Relative water distribution')
plt.ylabel('z position (A)')
plt.title('Relative distribution of waters')
plt.show()
