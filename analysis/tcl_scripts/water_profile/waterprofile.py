#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#get data
#df1 = pd.read_csv('ahelix_water_density.txt',delim_whitespace=True,names=['frame','z'])
#df2 = pd.read_csv('bhelix_water_density.txt',delim_whitespace=True,names=['frame','z'])
#df3 = pd.read_csv('interhelix_water_density.txt',delim_whitespace=True,names=['frame','z'])

#df = pd.concat([df1,df2,df3]).drop_duplicates().reset_index(drop=True)

df = pd.read_csv('pore_water_density.txt',delim_whitespace=True,names=['frame','z'])

#plot data
sns.distplot(df['z'],vertical=True)
#sns.distplot(df1['z'])
#sns.distplot(df2['z'])
#sns.distplot(df3['z'])
plt.title('Water Density Across Pore')
plt.xlabel('Distribution')
plt.ylabel('z position (A)')
plt.show()
