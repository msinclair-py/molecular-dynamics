#!/usr/bin/env python
from .funcs import (CoMDist, ContactFrequency, 
                             DeltaCOM, RadiusofGyration)
import gc
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.contacts import Contacts
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
import os
from typing import Union

class SimulationAnalyzer:
    def __init__(self, path: str, system: str, data_path: str='sim_data', 
                 stride: int=100):
        self.path = path
        self.system = system
        self.data_path = data_path
        self.stride = stride
        self.u = self.build_universe()

    def build_universe(self):
        pdb = f'{self.path}/{self.system}/protein.pdb'
        dcd = f'{self.path}/{self.system}/sim.dcd'
        try:
            u = mda.Universe(pdb, dcd, in_memory=True, in_memory_step=self.stride)
        except OSError:
            return 0

        return u

    def analyze(self):
        analyses = [self.binderRMSD, self.nativeContacts, self.deltaCoM,
                    self.comDist, self.rog, self.contactProb]
        dirs = ['rmsd_ppi', 'contacts_ppi', 'com_ppi', 
                'cdist_ppi', 'rog_ppi', 'cont_prob_ppi']
        sufs = [None, None, 'coms', 
                'raw', None, 'raw']

        for a, d, s in zip(analyses, dirs, sufs):
            if not os.path.exists(self.get_output(d, suffix=s)):
                a()

    def binderRMSD(self):
        R = RMSD(self.u, self.u, select='chainID A and backbone', 
                 groupselections=['chainID B and backbone'])
        R.run()

        np.save(self.get_output('rmsd_ppi'), R.rmsd[:, 3])

    def nativeContacts(self):
        sel_text = 'segid A and not type H'
        ref_text = 'segid B and not type H'
        sel = self.u.select_atoms(sel_text)
        ref = self.u.select_atoms(ref_text)

        ca1 = Contacts(self.u, select=(sel_text, ref_text),
                       refgroup=(sel, ref), radius=5, 
                       method='soft_cut')
        ca1.run()

        np.save(self.get_output('contacts_ppi'), ca1.timeseries)

    def deltaCoM(self):
        ref = self.u.select_atoms('segid B and not type H')

        dc = DeltaCOM(ref, residence=False)
        dc.run()

        np.save(self.get_output('com_ppi', suffix='coms'), dc.coms)

    def comDist(self):
        sel = self.u.select_atoms('segid A and backbone')
        ref = self.u.select_atoms('segid B and backbone')

        cd = CoMDist(sel, ref)
        cd.run()

        np.save(self.get_output('cdist_ppi', suffix='raw'), cd.com_dists)

    def rog(self):
        sel = self.u.select_atoms('segid B and name CA')
        
        rg = RadiusofGyration(sel)
        rg.run()

        np.save(self.get_output('rog_ppi'), rg.results)

    def contactProb(self):
        binder = self.u.select_atoms('segid B and name CA')
        target = self.u.select_atoms('segid A and name CA')

        cf = ContactFrequency(binder, target)
        cf.run()

        np.save(self.get_output('cont_prob_ppi', suffix='raw'), cf.contacts)
        np.save(self.get_output('cont_prob_ppi', suffix='prob'), cf.contact_probabilities)

    def get_output(self, d, suffix: Union[None, str]=None):
        if suffix is not None:
            ext = f'{suffix}.npy'
        else:
            ext = 'npy'

        return f'{self.data_path}/{d}/{self.system}.npy'

    def clean(self):
        del self.u
        gc.collect()

    @property
    def length(self):
        return len(self.u.trajectory)
