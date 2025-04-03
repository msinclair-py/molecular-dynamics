from collections import defaultdict
import json
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.util import convert_aa_code
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Callable, Dict, List, Tuple, Union

PathLike = Union[Path, str]
Results = Dict[str, Dict[str, float]]

class PPInteractions:
    def __init__(self, 
                 top: PathLike, 
                 traj: PathLike,
                 out: PathLike,
                 cov_cutoff: Tuple[float]=(11., 13.),
                 sb_cutoff: float=6.,
                 hbond_cutoff: float=3.5,
                 hbond_angle: float=30.,
                 hydrophobic_cutoff: float=8.,
                 plot: bool=True):
        self.u = mda.Universe(top, traj)
        self.n_frames = len(self.u.trajectory)
        self.out = out
        self.cov_cutoff = cov_cutoff
        self.sb = sb_cutoff
        self.hb_d = hbond_cutoff
        self.hb_a = hbond_angle * 180 / np.pi
        self.hydr = hydrophobic_cutoff
        self.plot = plot

    def run(self) -> None:
        """
        Main function that runs the workflow. Obtains a covariance matrix,
        screens for close interactions, evaluates each pairwise interaction
        for each amino acid and report the contact probability of each.
        """
        cov = self.get_covariance()
        positive, negative = self.interpret_covariance(cov)
        
        results = {'positive': {}, 'negative': {}}
        for res1, res2 in positive:
            data = self.compute_interactions(res1, res2)
            results['positive'].update(data)

        for res1, res2 in negative:
            data = self.compute_interactions(res1, res2)
            results['negative'].update(data)

        self.save(results)

        if self.plot:
            self.plot_results(results)

    def compute_interactions(self,
                             res1: int,
                             res2: int) -> Results:
        """
        Ingests two resIDs, generates MDAnalysis AtomGroups for each, identifies
        relevant non-bonded interactions (HBonds, saltbridge, hydrophobic) and
        computes each. Returns a Dict containing the proportion of simulation time
        that each interaction is engaged.
        """
        grp1 = self.u.select_atoms(f'chainID A and resid {res1}')
        grp2 = self.u.select_atoms(f'chainID B and resid {res2}')
        r1 = convert_aa_code(grp1.resnames[0])
        r2 = convert_aa_code(grp2.resnames[0])
        name = f'A_{r1}{res1}-B_{r2}{res2}'

        data = {name: {label: 0. for label in ['hydrophobic', 'hbond', 'saltbridge']}}
        function_calls, labels = self.identify_interaction_type(grp1.resnames[0], grp2.resnames[0])
        for call, label in zip(function_calls, labels):
            data[name][label] = call(grp1, grp2)

        return data

    def get_covariance(self) -> np.ndarray:
        """
        Loop over all C-alpha atoms and compute the positional
        covariance using the functional form:
            C = <(R1 - <R1>)(R2 - <R2>)T>
        where each element corresponds to the ensemble average movement
            C_ij = <deltaR_i * deltaR_j>
        with the magnitude being the strength of correlation and the sign
        corresponding to positive and negative correlation respectively.
        """
        p1_ca = self.u.select_atoms('chainID A and name CA')
        N = p1_ca.n_residues

        p2_ca = self.u.select_atoms('chainID B and name CA')
        M = p2_ca.n_residues

        self.res_map(p1_ca, p2_ca)

        R1_avg = np.zeros((N, 3))
        R2_avg = np.zeros((M, 3))

        for ts in self.u.trajectory:
            R1_avg += p1_ca.positions
            R2_avg += p2_ca.positions

        R1_avg /= self.n_frames
        R2_avg /= self.n_frames
        
        C = np.zeros((N, M))

        for ts in self.u.trajectory:
            R1 = p1_ca.positions
            R2 = p2_ca.positions

            dR1 = R1 - R1_avg
            dR2 = R2 - R2_avg

            for i in range(N):
                for j in range(M):
                    C[i, j] += np.dot(dR1[i], dR2[j])

        C /= self.n_frames
        
        for i in range(N):
            for j in range(M):
                dist = np.linalg.norm(R1_avg[i] - R2_avg[j])
                if C[i, j] > 0:
                    if dist > self.cov_cutoff[0]:
                        C[i, j] = 0.
                elif dist > self.cov_cutoff[1]:
                    C[i, j] = 0.

        return C

    def res_map(self,
                ag1: mda.AtomGroup,
                ag2: mda.AtomGroup) -> None:
        """
        Map covariance matrix indices to AtomGroup resIDs so that we are
        examining the correct pairs of residues.
        """
        mapping = {'ag1': {}, 'ag2': {}}
        for i, resid in enumerate(ag1.resids):
            mapping['ag1'][i] = resid

        for i, resid in enumerate(ag2.resids):
            mapping['ag2'][i] = resid

        self.mapping = mapping

    def interpret_covariance(self,
                             cov_mat: np.ndarray) -> Tuple[Tuple[int, int]]:
        """
        Identify pairs of residues with positive or negative correlations.
        Returns a tuple comprised of pairs for each.
        """
        pos_corr = np.where(cov_mat > 0.)
        neg_corr = np.where(cov_mat < 0.)
       
        seen = set()
        positive = list()
        for i in range(len(pos_corr[0])):
            res1 = self.mapping['ag1'][pos_corr[0][i]]
            res2 = self.mapping['ag2'][pos_corr[1][i]]
            if (res1, res2) not in seen:
                positive.append((res1, res2))
                seen.add((res1, res2))
                seen.add((res2, res1))

        negative = list()
        for i in range(len(neg_corr[0])):
            res1 = self.mapping['ag1'][neg_corr[0][i]]
            res2 = self.mapping['ag2'][neg_corr[1][i]]
            if (res1, res2) not in seen:
                negative.append((res1, res2))
                seen.add((res1, res2))
                seen.add((res2, res1))

        return positive, negative

    def identify_interaction_type(self,
                                  res1: str,
                                  res2: str) -> List[Callable]:
        """
        Identifies what analyses to compute for a given pair of protein
        residues (i.e. hydrophobic interactions, hydrogen bonds, saltbridges).
        """
        int_types = {
            'TYR': {'funcs': [self.analyze_hbond], 'label': ['hbond']}, 
            'HIS': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'HID': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'HIE': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'SER': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'THR': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'ASN': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'GLN': {'funcs': [self.analyze_hbond], 'label': ['hbond']},
            'ASP': {'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                    'label': ['hbond', 'saltbridge']},
            'GLU': {'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                    'label': ['hbond', 'saltbridge']},
            'LYS': {'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                    'label': ['hbond', 'saltbridge']},
            'ARG': {'funcs': [self.analyze_hbond, self.analyze_saltbridge],
                    'label': ['hbond', 'saltbridge']},
            'HIP': {'funcs': [self.analyze_hbond, self.analyze_saltbridge], 
                    'label': ['hbond', 'saltbridge']},
        }
        
        funcs = defaultdict(lambda: [[], []])
        for res, calls in int_types.items():
            funcs[res] = [calls['funcs'], calls['label']]

        functions = [self.analyze_hydrophobic]
        labels = ['hydrophobic']
        for func, lab in zip(*funcs[res1]):
            if func in funcs[res2][0]:
                functions.append(func)
                labels.append(lab)

        return functions, labels

    def analyze_saltbridge(self,
                           res1: mda.AtomGroup,
                           res2: mda.AtomGroup) -> float:
        """
        Uses a simple distance cutoff to highlight the occupancy of 
        saltbridge between two residues. Returns the fraction of
        simulation time spent engaged in saltbridge.
        """
        pos = ['LYS', 'ARG']
        neg = ['ASP', 'GLU']
        name1 = res1.resnames[0]
        name2 = res2.resnames[0]
        if name1 not in pos + neg:
            return 0.
        elif name2 not in pos + neg:
            return 0.
        elif name1 in pos and name2 in pos:
            return 0.
        elif name1 in neg and name2 in neg:
            return 0.
        
        atom_names = ['NZ', 'NH1', 'NH2', 'OD1', 'OD2', 'OE1', 'OE2']

        grp1 = self.u.select_atoms('resname DUMMY')
        for atom in res1.atoms:
            if atom.name in atom_names:
                grp1 += atom

        grp2 = self.u.select_atoms('resname DUMMY')
        for atom in res2.atoms:
            if atom.name in atom_names:
                grp2 += atom
        
        n_frames = 0
        for ts in self.u.trajectory:
            dist = np.linalg.norm(grp1.positions - grp2.positions)
            if dist < self.sb:
                n_frames += 1
        
        return n_frames / self.n_frames

    def analyze_hbond(self,
                      res1: mda.AtomGroup,
                      res2: mda.AtomGroup) -> float:
        """
        Identifies all potential donor/acceptor atoms between two
        residues. Culls this list based on distance array across simulation
        and then evaluates each pair over the trajectory utilizing a
        distance and angle cutoff.
        """
        donors, acceptors = self.survey_donors_acceptors(res1, res2)

        n_frames = 0
        for ts in self.u.trajectory:
            n_frames += self.evaluate_hbond(donors, acceptors)

        return n_frames / self.n_frames

    def analyze_hydrophobic(self,
                            res1: mda.AtomGroup,
                            res2: mda.AtomGroup) -> float:
        h1 = self.u.select_atoms('resname DUMMY')
        h2 = self.u.select_atoms('resname DUMMY')

        for atom in res1.atoms:
            if 'C' in atom.type:
                h1 += atom

        for atom in res2.atoms:
            if 'C' in atom.type:
                h2 += atom

        n_frames = 0
        for ts in self.u.trajectory:
            da = distance_array(h1, h2)
            if np.min(da) < self.hydr:
                n_frames += 1

        return n_frames / self.n_frames

    def survey_donors_acceptors(self,
                                res1: mda.AtomGroup,
                                res2: mda.AtomGroup) -> Tuple[mda.AtomGroup]:
        """
        First pass distance threshhold to identify potential Hydrogen bonds.
        Should be followed by querying HBond angles but this serves to reduce
        our search space and time complexity. Only returns donors/acceptors which
        are within the distance cutoff in at least a single frame.
        """
        donors = self.u.select_atoms('resname DUMMY')
        acceptors = self.u.select_atoms('resname DUMMY')

        for atom in res1.atoms:
            if any([a in atom.type for a in ['O', 'N']]):
                if any(['H' in bond for bond in atom.bonded_atoms.types]):
                    donors += atom
                acceptors += atom
        
        for atom in res2.atoms:
            if any([a in atom.type for a in ['O', 'N']]):
                if any(['H' in bond for bond in atom.bonded_atoms.types]):
                    donors += atom
                acceptors += atom

        distances = distance_array(donors, acceptors)
        contacts = np.where(distances < self.hb_d)
        don_contacts = np.unique(contacts[0])
        acc_contacts = np.unique(contacts[1])

        return donors[don_contacts], acceptors[acc_contacts]

    def evaluate_hbond(sel,
                       donor: mda.AtomGroup,
                       acceptor: mda.AtomGroup) -> int:
        """
        Evaluates whether there is a defined hydrogen bond between any
        donor and acceptor atoms in a given frame. Must pass a distance
        cutoff as well as an angle cutoff. Returns early when a legal
        HBond is detected.
        """
        for d in donor.atoms:
            pos1 = d.position
            hpos = [atom.position for atom in d.bonded_atoms if 'H' in atom.type]
            for a in acceptor.atoms:
                pos3 = a.position

                if np.linalg.norm(pos3 - pos1) <= self.hb_d:
                    for pos2 in hpos:
                        v1 = pos2 - pos1
                        v2 = pos3 - pos2

                        v1 /= np.linalg.norm(v1)
                        v2 /= np.linalg.norm(v2)

                        if np.arccos(np.dot(v1, v2)) <= self.hb_a:
                            return 1

        return 0

    def save(self,
             results: Results) -> None:
        """
        Save results as a json file.
        """
        with open(self.out, 'w') as fout:
            json.dump(results, fout, indent=4)
    
    def plot_results(self,
                     results: Results) -> None:
        """
        """
        df = self.parse_results(results)
        
        plot = Path('plots')
        plot.mkdir(exist_ok=True)
        for cov_type in ['positive', 'negative']:
            for int_type in ['Hydrophobic', 'Hydrogen Bond', 'Salt Bridge']:
                data = df[(df['Covariance'] == cov_type) & (df[int_type] > 0.)]

                if not data.empty:
                    name = f'{cov_type.capitalize()}_Covariance_'
                    name += f'{"_".join(int_type.split(" "))}.png'

                    self.make_plot(
                        data,
                        int_type,
                        plot / name
                    )
    
    def parse_results(self, 
                      results: Results) -> pd.DataFrame:
        """
        Prepares results for plotting. Removes any entries which are
        all 0. and returns as a pandas DataFrame for easier plotting.
        """
        data_rows = []
        for cov_type, pair_dict in results.items():
            for pair, data in pair_dict.items():
                if any(val > 0. for val in data.values()):
                    row = {
                        'Residue Pair': pair,
                        'Hydrophobic': data['hydrophobic'],
                        'Hydrogen Bond': data['hbond'],
                        'Salt Bridge': data['saltbridge'],
                        'Covariance': cov_type,
                    }

                    data_rows.append(row)

        return pd.DataFrame(data_rows)
    
    def make_plot(self, 
                  data: pd.DataFrame,
                  column: str,
                  name: PathLike,
                  fs: int=15) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        sns.barplot(data=data, x='Residue Pair', y=column, ax=ax)
        
        ax.set_xlabel('Residue Pair', fontsize=fs)
        ax.set_ylabel('Probability', fontsize=fs)
        ax.set_title(column, fontsize=fs+2)
        ax.tick_params(labelsize=fs)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(str(name), dpi=300)
