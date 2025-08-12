import openmm
from openmm.app import AmberPrmtopFile
import MDAnalysis as mda
from numba import njit
import numpy as np
from pathlib import Path
from typing import Union

OptPath = Union[Path, str, None]
PathLike = Union[Path, str]

@njit
def unravel_index(n1: int, 
                  n2: int) -> tuple[np.ndarray, np.ndarray]:
    """
    
    """
    a, b = np.empty((n1, n2), dtype=np.int32), np.empty((n1, n2), dtype=np.int32)
    for i in range(n1):
        for j in range(n2):
            a[i,j], b[i,j] = i, j
    return a.ravel(),b.ravel()

@njit
def _dist_mat(xyz1: np.ndarray, 
              xyz2: np.ndarray) -> np.ndarray:
    """

    """
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
def dist_mat(xyz1: np.ndarray, 
             xyz2: np.ndarray) -> np.ndarray:
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return _dist_mat(xyz1, xyz2).reshape(n1, n2)

@njit
def electrostatic(distance,
                  charge_i, 
                  charge_j):
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
    #     Avogadro = 6.022e23 molecules/mol
    #     e- to Coloumb = 1.602e-19 C/e-
    #     nm to m = 1e-9 m/nm
    #     1/(4\pi\epsilon_0) = 8.988e9 J*m/C^2
    
    solvent_dielectric = 78.5
    # calculate energy
    if distance > 1.:
        energy = 0.
    else:
        r = distance * 1e-9
        r_cutoff = 1. * 1e-9
        k_rf = 1 / (r_cutoff ** 3) * (solvent_dielectric - 1) / (2 * solvent_dielectric + 1)
        c_rf = 1 / r_cutoff * (3 * solvent_dielectric) / (2 * solvent_dielectric + 1)

        outer_term = 8.988e9 * (charge_i * 1.602e-19) * (charge_j * 1.602e-19)
        energy = outer_term * (1 / r + k_rf * r ** 2 - c_rf) * 6.022e23
    return energy / 1000 # J -> kJ

@njit
def electrostatic_sum(distances,
                      charge_is, 
                      charge_js):
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
                  sigma_i, 
                  sigma_j,
                  epsilon_i, 
                  epsilon_j):
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
                      sigma_is, 
                      sigma_js,
                      epsilon_is, 
                      epsilon_js):
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
                                    epsilon_is[i], epsilon_js[j])
    return energy

@njit
def fingerprints(xyzs, 
                 charges, 
                 sigmas, 
                 epsilons,
                 target_resmap, 
                 binder_inds):
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
    for i in range(n_target_residues):
        dists = dist_mat(xyzs[target_resmap[i]], xyzs[binder_inds])
        es_fingerprint[i] = electrostatic_sum(dists,
                                              charges[target_resmap[i]],
                                              charges[binder_inds])
        lj_fingerprint[i] = lennard_jones_sum(dists,
                                              sigmas[target_resmap[i]],
                                              sigmas[binder_inds],
                                              epsilons[target_resmap[i]],
                                              epsilons[binder_inds])
    return lj_fingerprint, es_fingerprint

class Fingerprinter:
    """
    Calculates interaction energy fingerprint between target and binder chains. 
    
    Arguments:
        topology (PathLike): Path to topology file.
        trajectory (OptPath): Defaults to None. If not None, should be a path to a
            trajectory file or coordinates file.
        target_selection (str): Defaults to 'segid A'. Any MDAnalysis selection
            string that encompasses target.
        binder_selection (str | None): Defaults to None. If None, binder is defined
            as anything that is not the target. Otherwise should be an MDAnalysis
            selection string that encompasses the binder.
        out_path (OptPath): Defaults to None. If provided will be where outputs are
            saved to, otherwise the topology parent path is used.
        out_name (str | None): Defaults to None. If None, output file will be called
            'fingerprint.npz'.
            
    Usage:
        m = Fingerprinter(*args)
        m.run()
        m.save()
    """
    def __init__(self,
                 topology: PathLike,
                 trajectory: OptPath=None,
                 target_selection: str = 'segid A',
                 binder_selection: str | None = None,
                 out_path: OptPath = None,
                 out_name: str | None = None):
        self.topology = Path(topology)
        self.trajectory = Path(trajectory) if trajectory is not None else trajectory
        self.target_selection = target_selection

        if binder_selection is not None:
            self.binder_selection = binder_selection
        else:
            self.binder_selection = f'not {target_selection}'

        if out_path is None:
            path = self.topology.parent
        else:
            path = Path(out_path)

        if out_name is None:
            self.out = path / 'fingerprint.npz'
        else:
            self.out = path / out_name

    def assign_nonbonded_params(self) -> None:
        # build openmm system
        system = AmberPrmtopFile(self.topology).createSystem()

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
        """
        Loads topology into MDAnalysis universe object. Can take either a PDB
        or an AMBER prmtop. If trajectory was not specified, will look for either
        inpcrd or rst7 with the same path and stem as the topology file.

        Returns:
            None
        """
        if self.topology.suffix == '.pdb':
            self.u = mda.Universe(self.topology)
        else:
            if self.trajectory is not None:
                coordinates = self.trajectory
            elif self.topology.with_suffix('.inpcrd').exists():
                coordinates = self.topology.with_suffix('.inpcrd')
            else:
                coordinates = self.topology.with_suffix('.rst7')

            self.u = mda.Universe(self.topology, coordinates)

    def assign_residue_mapping(self) -> None:
        """
        Map each residue index (1-based) to corresponding atom indices.

        Returns:
            None
        """
        target = self.u.select_atoms(self.target_selection)
        self.target_resmap = [residue.atoms.ix for residue in target.residues]
        self.target_inds = np.concatenate(self.target_resmap)


        binder = self.u.select_atoms(self.binder_selection)
        self.binder_resmap = [residue.atoms.ix for residue in binder.residues]
        self.binder_inds = np.concatenate(self.binder_resmap)

    def iterate_frames(self) -> None:
        """
        Runs calculations over each frame.

        Returns:
            None
        """
        self.target_fingerprint = np.zeros((
            len(self.u.trajectory), len(self.target_resmap), 2
        ))

        self.binder_fingerprint = np.zeros((
            len(self.u.trajectory), len(self.binder_resmap), 2
        ))

        for i, ts in enumerate(self.u.trajectory):
            self.calculate_fingerprints(i)

    def calculate_fingerprints(self,
                               frame_index: int) -> None:
        """
        Calculates fingerprints for a given frame of the trajectory.

        Arguments:
            frame_index (int): Frame index, since frame number and index might be
                discontinuous.
        Returns:
            None
        """
        positions = self.u.atoms.positions * .1 # convert to nm
        self.target_fingerprint[frame_index] = np.vstack(
            fingerprints(
                positions,
                self.charges,
                self.sigmas, self.epsilons,
                self.target_resmap, self.binder_inds
            )
        ).T

        self.binder_fingerprint[frame_index] = np.vstack(
            fingerprints(
                positions,
                self.charges,
                self.sigmas, self.epsilons,
                self.binder_resmap, self.target_inds
            )
        ).T
    
    def run(self) -> None:
        """
        Main logic. Obtains parameters, loads PDB and then
        iterates through trajectory to obtain fingerprints.

        Returns:
            None
        """
        self.assign_nonbonded_params()
        self.load_pdb()
        self.assign_residue_mapping()
        self.iterate_frames()

    def save(self) -> None:
        """
        Saves data to an npz stack.

        Returns:
            None
        """
        np.savez(self.out, 
                 target=self.target_fingerprint, 
                 binder=self.binder_fingerprint)
