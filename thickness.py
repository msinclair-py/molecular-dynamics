#!/usr/bin/env python
import sys, os, glob
from os.path import dirname, abspath
 
outfile = open('thick.txt', 'w')

###Function to calculate mean 
def mean(numbers):
   return float(sum(numbers)) / max(len(numbers), 1)

###Calculated thickness of average z positions
with open('netavg.pdb', 'r') as infile:
   pos = []
   neg = []   
   p = []
   n = []   
   lst = []
   for line in infile:
      if "CRYST1" in line:
         continue
      elif "END" in line:
         continue
      else:
         lst.append(line)

   for line in lst:
      search = line.split()
      if float(search[7]) > 0:
         pos.append(search[7])
      elif float(search[7]) < 0:
         neg.append(search[7])

   for index, item in enumerate(pos):
      pos[index] = float(item)
   
   for index, item in enumerate(neg):
      neg[index] = float(item)

   p = mean(pos)
   n = mean(neg)
   
   thick = p + abs(n)
   outfile.write("Average thickness is: " + str(thick))



###Calculate each thickness ring around protein
d = dirname(dirname(abspath(__file__)))
pdbCounter = len(glob.glob1(d, "*.pdb"))

for i in range(1,int(pdbCounter)):
   filename = "ring%d.pdb" % i
   with open(filename, 'r') as infile: 
      pos = []
      neg = []
      p = []
      n = []
      lst = []
      for line in infile:
         if "CRYST1" in line:
            continue
         elif "END" in line:
            continue
         else:
            lst.append(line)

      for line in lst:
         search = line.split()
         if float(search[7]) > 0:
            pos.append(search[7])
         else:
            neg.append(search[7])

      for index, item in enumerate(pos):
         pos[index] = float(item)

      for index, item in enumerate(neg):
         neg[index] = float(item)

      p = mean(pos)
      n = mean(neg)

      ring = p + abs(n)
      outfile.write("\nRing %r average thickness is: " % i + str(ring))

 
