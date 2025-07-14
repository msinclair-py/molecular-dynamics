import openmm
import numpy as np
import mdtraj as md
from numba import njit, prange
from ..simulate import Minimizer
from typing import List

@njit
def unravel_index(n1, n2):
    a, b = np.empty((n1, n2), dtype=np.int32), np.empty((n1, n2), dtype=np.int32)
    for i in range(n1):
        for j in range(n2):
            a[i,j], b[i,j] = i, j
    return a.ravel(),b.ravel()

@njit
def dist_mat_(xyz1, xyz2):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    dist_mat = np.zeros((n1 * n2))
    i, j = unravel_index(n1, n2)
    for k in range(n1 * n2):
        dr = xyz1[i[k]] - xyz2[j[k]]
        for ri in range(ndim):
            dist_mat[k] += np.square(dr[ri])
    return np.sqrt(dist_mat)

@njit
def dist_mat(xyz1, xyz2):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_mat_(xyz1, xyz2).reshape(n1, n2)

@njit
def electrostatic(distance,
                  charge_i, charge_j):
    """
    Calculate electrostatic energy between two particles.
    Cutoff at 12 Angstrom without switching.

    Parameters:
        distance (float): distance between particles i and j (nm)
        charge_i (float): charge of particle i (e-)
        charge_j (float): charge of particle j (e-)

    Returns:
        energy (float): Electrostatic energy between particles (kJ/mol)
    """
    # conversion factors:
    #     Avogadro = 1.626e23
    #     e- to Coloumb = 1.602e-19
    #     nm to m = 1e-9
    #     1/(4\pi\epsilon_0) = 8.988e9

    # calculate energy
    if distance > 1.2:
        energy = 0.
    else:
        numerator = 8.988e9 * (charge_i * 1.602e-19) * (charge_j * 1.602e-19)
        denominator = distance * 1e-9
        energy = 1.626e23 * numerator / denominator
    return energy

@njit
def electrostatic_sum(distances,
                      charge_is, charge_js):
    """
    Calculate sum of all electrostatic interactions between two
    sets of particles.

    Parameters:
        distances (np.ndarray): distances between particles,
            shape: (len(charge_is),len(charge_js))
        charge_is (np.ndarray): group i charges
        charge_js (np.ndarray): group j charges
    """
    n = distances.shape[0]
    m = distances.shape[1]
    energy = 0.
    for i in range(n):
        for j in range(m):
            energy += electrostatic(distances[i,j],
                                    charge_is[i],
                                    charge_js[j])
    return energy

@njit
def lennard_jones(distance, 
                  sigma_i, sigma_j,
                  epsilon_i, epsilon_j):
    """
    Calculate LJ energy between two particles.
    Cutoff at 12 Angstrom without switching.

    Parameters
    ----------
    distance (float): distance between particles i and j (nm)
    sigma_i (float): sigma parameter for particle i (nm)
    sigma_j (float): sigma parameter for particle j (nm)
    epsilon_i (float): epsilon parameter for particle i (kJ/mol)
    epsilon_j (float): epsilon parameter for particle j (kJ/mol)

    Returns: energy (float): LJ interaction energy (kJ/mol)
    """
    if distance > 1.2:
        energy = 0.
    else:
        # use combination rules to solve for epsilon and sigma
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        epsilon_ij = np.sqrt(epsilon_i * epsilon_j) 
    
        # calculate energy
        sigma_r = sigma_ij / distance
        sigma_r_6 = sigma_r ** 6
        sigma_r_12 = sigma_r_6 ** 2
        energy = 4. * epsilon_ij * (sigma_r_12 - sigma_r_6)
    return energy

@njit
def lennard_jones_sum(distances,
                      sigma_is, sigma_js,
                      epsilon_is, epsilon_js):
    """
    Calculate sum of all LJ interactions between two sets of
    particles.                                       

    Parameters:
        distances (np.ndarray): distances between particles, 
            shape: (len(sigma_is),len(sigma_js))
        sigma_is (np.ndarray): group i sigma parameters
        sigma_js (np.ndarray): group j sigma parameters
        epsilon_is (np.ndarray): group i epsilon parameters
        epsilon_js (np.ndarray): group j epsilon parameters
    """
    n = distances.shape[0]
    m = distances.shape[1]
    energy = 0.
    for i in range(n):
        for j in range(m):
            energy += lennard_jones(distances[i,j], 
                                    sigma_is[i], sigma_js[j],
                                    epsilon_is[i], sigma_js[j])
    return energy

@njit(parallel=True)
def fingerprints(xyzs, charges, sigmas, epsilons,
                 target_resmap, binder_inds):
    """
    Calculates electrostatic fingerprint.
    ES energy between each target residue and all binder residues.

    Returns:
        fingerprints: (np.ndarray, np.ndarray), 
            shape=(n_target_residues, n_target_residues)
    """
    n_target_residues = len(target_resmap)
    es_fingerprint = np.zeros((n_target_residues))
    lj_fingerprint = np.zeros((n_target_residues))
    for i in prange(n_target_residues):
        dists = dist_mat(xyzs[target_resmap[i]], xyzs[binder_inds])
        es_fingerprint[i] = electrostatic_sum(dists,
                                              charges[target_resmap[i]],
                                              charges[binder_inds])
        lj_fingerprint[i] = lennard_jones_sum(dists,
                                              sigmas[target_resmap[i]],
                                              sigmas[binder_inds],
                                              epsilons[target_resmap[i]],
                                              epsilons[binder_inds])
    return es_fingerprint, lj_fingerprint

class Fingerprinter:
    """
    Calculates interaction energy fingerprint between target and binder chains. 
    
    Inputs:
        pdb_file (str): path to pdb file
        target_resid_range (List[int]): inclusive range of residue indices (1-based) 
            defining target protein
        binder_resid_range (List[int]): inclusive range of residue indices (1-based) 
            defining binder protein
            
    Usage:
        m = Fingerprinter(*args)
        m.run()
        m.save()
    """
    def __init__(self,
                 pdb_file: str,
                 target_resid_range: List[int],
                 binder_resid_range: List[int]):
        self.pdb_file = pdb_file
        self.target_resid_range = target_resid_range
        self.binder_resid_range = binder_resid_range

    def assign_nonbonded_params(self) -> None:
        # build openmm system
        builder = Minimizer(topology=self.pdb_file)
        system = builder.load_pdb()

        # extract NB params
        nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        self.epsilons = np.zeros((system.getNumParticles()))
        self.sigmas = np.zeros((system.getNumParticles()))
        self.charges = np.zeros((system.getNumParticles()))
        for ind in range(system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(ind)
            self.charges[ind] = charge / charge.unit # elementary charge
            self.sigmas[ind] = sigma / sigma.unit # nm
            self.epsilons[ind] = epsilon / epsilon.unit # kJ/mol

    def load_pdb(self) -> None:
        # load with mdtraj
        self.traj = md.load(self.pdb_file)

    def assign_residue_mapping(self) -> None:
        # map each residue index (1-based) to corresponding atom indices
        target_resmap = []
        for resid in range(self.target_resid_range[0], 
                           self.target_resid_range[1] + 1):
            target_resmap.append(self.traj.top.select(f'residue {resid}'))
        self.target_resmap = target_resmap
        self.target_inds = np.concatenate(self.target_resmap)
        binder_resmap = []
        for resid in range(self.binder_resid_range[0], 
                           self.binder_resid_range[1] + 1):
            binder_resmap.append(self.traj.top.select(f'residue {resid}'))
        self.binder_resmap = binder_resmap
        self.binder_inds = np.concatenate(self.binder_resmap)

    def calculate_fingerprints(self) -> None:
        self.target_es_fingerprint, self.target_lj_fingerprint = \
            fingerprints(
                self.traj.xyz[0], # assume only one frame
                self.charges,
                self.sigmas, self.epsilons,
                self.target_resmap, self.binder_inds)

        self.binder_es_fingerprint, self.binder_lj_fingerprint = \
            fingerprints(
                self.traj.xyz[0], # assume only one frame
                self.charges,
                self.sigmas, self.epsilons,
                self.binder_resmap, self.target_inds)
    
    def run(self) -> None:
        self.assign_nonbonded_params()
        self.load_pdb()
        self.assign_residue_mapping()
        self.calculate_fingerprints()

    def save(self, out_file_prefix: str = '') -> None:
        sep = '_' if out_file_prefix else ''
        np.save(out_file_prefix + sep + 'target_energies.npy',
                np.vstack([self.target_es_fingerprint,
                           self.target_lj_fingerprint]).T)
        np.save(out_file_prefix + sep + 'binder_energies.npy',
                np.vstack([self.binder_es_fingerprint,
                           self.binder_lj_fingerprint]).T)
