#!/usr/bin/env python
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

class DeltaCOM(AnalysisBase):
    """
    Computes the change in CoM in a sliding window, saving the delta distances into
    an array. Then loops over the array to determine which if any resident binding
    events have occurred, how long they occurred for and in what order they occurred.
    """
    def __init__(self, sel: mda.AtomGroup, sliding_window: int=1, residence: bool=True,
                 residence_distance_cutoff: float=0.5, min_residence: int=5):
        super().__init__(sel.universe.trajectory)
        self.sel = sel
        self.window = 1
        self.residence = residence
        self.cutoff = residence_distance_cutoff
        self.min_res = min_residence

    def _prepare(self):
        self.coms = np.zeros((len(self._trajectory), 3))
        self.delta_coms = np.zeros((len(self._trajectory)))

    def _single_frame(self):
        fr = self._frame_index
        self.coms[fr, :] = self.sel.center_of_mass()
        self.delta_coms[fr] = self.distance(fr)

    def _conclude(self):
        if self.residence:
            self.residences = []
            length = 1
            for delta in self.delta_coms[1:]:
                if delta <= self.cutoff:
                    length += 1
                else:
                    self.residences.append(length)
                    length = 1

            self.residences = np.array(self.residences)
            self.residences = self.residences[np.where(self.residences >= self.min_res)]

    def distance(self, frame: int):
        """
        Euclidean distance is just L2 norm, use numpy.linalg.norm to
        compute this efficiently, or if its the first frame and we throw
        an IndexError return a NaN.
        """
        try:
            return np.linalg.norm(self.coms[frame, :] - self.coms[frame-self.window, :])
        except IndexError:
            return np.nan


class CoMDist(AnalysisBase):
    def __init__(self, target: mda.AtomGroup, binder: mda.AtomGroup):
        super().__init__(target.universe.trajectory)
        self.target = target
        self.binder = binder

    def _prepare(self):
        self.com_dists = np.zeros((len(self._trajectory)))

    def _single_frame(self):
        pos1 = self.target.center_of_mass()
        pos2 = self.binder.center_of_mass()

        self.com_dists[self._frame_index] = np.linalg.norm(pos1 - pos2)

    def _conclude(self):
        self.rel_dists = self.com_dists - self.com_dists[0]


class RadiusofGyration(AnalysisBase):
    def __init__(self, atomgroup):
        super().__init__(atomgroup.universe.trajectory)
        self.ag = atomgroup
        self.masses = atomgroup.masses
        self.total_mass = np.sum(self.masses)

    def _prepare(self):
        self.results = np.zeros((self.n_frames, 4))

    def _single_frame(self):
        rogs = self.radgyr(self.ag, self.masses, 
                           total_mass=self.total_mass)
        self.results[self._frame_index] = rogs

    def _conclude(self):
        pass

    @staticmethod
    def radgyr(atomgroup, masses, total_mass=None):
        coordinates = atomgroup.positions
        center_of_mass = atomgroup.center_of_mass()

        ri_sq = (coordinates-center_of_mass)**2
        sq = np.sum(ri_sq, axis=1)
        sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
        sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
        sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

        sq_rs = np.array([sq, sq_x, sq_y, sq_z])

        rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
        return np.sqrt(rog_sq)


class ContactFrequency(AnalysisBase):
    def __init__(self, binder, target, cutoff=5.):
        super().__init__(binder.universe.trajectory)
        self.binder = binder
        self.target = target
        self.cutoff = cutoff

    def _prepare(self):
        self.distance_array = np.zeros((self.n_frames, self.binder.n_atoms, self.target.n_atoms))
        self.contact_probabilities = np.zeros((self.n_frames))

    def _single_frame(self):
        distance_array(self.binder, self.target, 
                       result=self.distance_array[self._frame_index])
        
    def _conclude(self):
        contacts = np.where(self.distance_array < self.cutoff)
        self.contacts = np.unique(np.column_stack(contacts[:2]), axis=0)

        frames, counts = np.unique(self.contacts[:, 0], return_counts=True)
        for (frame, count) in zip(frames, counts):
            self.contact_probabilities[frame] = count

        self.contact_probabilities /= self.binder.n_atoms

