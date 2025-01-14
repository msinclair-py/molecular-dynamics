#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

numres = 91 
#98 reduced pore selection 
# reduced entrance helixes 141 
# 45 in pore selection and 96 in outer helices selection <- no longer accurate

numframes = 26

dframe = pd.read_csv('residences.txt',delim_whitespace=True,names=['frame','resid','counts'])
dat = np.zeros((numres,2))

for i in range(numres):
	df = dframe.iloc[i::numres]
	df.reset_index(drop=True,inplace=True)
	dat[i] = [df.iloc[0,1],df['counts'].sum()]

final = pd.DataFrame(data=dat,columns=['resid','counts'],dtype=int)
final['counts'] /= numframes
final.sort_values(by=['resid'],inplace=True)

eh1 = final.iloc[:26]
eh2 = final.iloc[26:53]
pore = final.iloc[53:]

sns.lineplot(x='resid',y='counts',data=eh1,label='Entrance Helix 1')
sns.lineplot(x='resid',y='counts',data=eh2,label='Entrance Helix 2')
sns.lineplot(x='resid',y='counts',data=pore,label='Central Pore')

plt.xlabel('ResID')
plt.ylabel('Percent Residence Over Trajectory')
plt.xticks(np.arange(325,476,10))
plt.legend(loc='best')

plt.title('Percent Ion Residence per Residue')
plt.savefig('IonResidence.png')
plt.show()
