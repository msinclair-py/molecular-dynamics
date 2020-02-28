#!/usr/bin/env python
import sys, os, glob
from os.path import dirname, abspath
import numpy as np

infiles = ['/Scr/msincla01/YidC_Membrane_Simulation/Analysis/Local_Hydration/periplasm%i.txt'%i for i in range(8)]

rawdata = []
for i in range(len(infiles)):
        rawdata.append(np.genfromtxt(infiles[i]))
data = np.zeros((rawdata[0].shape))
data[:,0] = rawdata[0][:,0]
data[:,1] = rawdata[0][:,1]
for i in range(len(infiles)):
        data[:,2] += rawdata[i][:,2]
data[:,2] /= len(infiles)

infiles2 = ['/Scr/msincla01/YidC_Membrane_Simulation/Analysis/Local_Hydration/cytoplasm%i.txt'%i for i in range(8)]

rawdata2 = []
for i in range(len(infiles2)):
        rawdata2.append(np.genfromtxt(infiles2[i]))
data2 = np.zeros((rawdata2[0].shape))
data2[:,0] = rawdata2[0][:,0]
data2[:,1] = rawdata2[0][:,1]
for i in range(len(infiles)):
        data2[:,2] += rawdata2[i][:,2]
data2[:,2] /= len(infiles)

np.savetxt("periplasm_density.txt",data,fmt="%s")

np.savetxt("cytoplasm_density.txt",data2,fmt="%s")
