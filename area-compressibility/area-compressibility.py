#!/usr/bin/env python
import numpy as np
from decimal import *

def area_compressibility(a, v):
   #kb must be converted to A which will be cancelled out ultimately
   #m2 to A2 multiplies by 1e20
   kb = np.float64(1.38064852e-3)
   T = 310
   A = a
   variance = v
   ka = kb * T * A / variance
   return ka

with open('avgarea.txt', 'r') as infile:
   arr = []
   add = 0
   var = 0
   fl = 0

   for line in infile:
      arr = np.append(arr, float(line.split()[1]))

   add = np.mean(arr, dtype = np.float64)
   var = np.var(arr, dtype = np.float64)
   Ka = area_compressibility(add, var)

with open('area.compressibility.txt', 'w') as outfile:
   outfile.write("Average area is %f\nVariance is %f\nArea Compressibility is %s"%(add,var,Ka))
