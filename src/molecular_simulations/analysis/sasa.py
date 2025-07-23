import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core import groups
from MDAnalysis.guesser.tables import vdwradii
import numpy as np
from scipy.spatial import KDTree
import warnings

warnings.filterwarnings('ignore')

class SASA(AnalysisBase):
    """
    Computes solvent-accessible surface area using the Shrake-Rupley algorithm.
    Code is adapted from the biopython and mdtraj implementations as well as an
    unmerged PR from MDAnalysis. Implemented as an instance of AnalysisBase to 
    support ease of deployment and baked-in parallelism.

    Arguments:
        ag (mda.AtomGroup): AtomGroup for which to perform SASA.
        probe_radius (float): Defaults to 1.4Å. This is pretty standard and should not
            be altered in most cases.
        n_points (int): Defaults to 256. Number of points to place on each sphere, where
            a higher number is more costly but accurate.
    """
    def __init__(self, 
                 ag: mda.AtomGroup,
                 probe_radius: float=1.4,
                 n_points: int=256,
                 **kwargs):
        if isinstance(ag, groups.UpdatingAtomGroup):
            raise TypeError('UpdatingAtomGroups are not valid for SASA!')
        
        super(SASA, self).__init__(ag.universe.trajectory, **kwargs)
        
        if not hasattr(ag, 'elements'):
            raise ValueError('Cannot assign atomic radii:'
                             'Universe has no `elements` property!')

        self.ag = ag
        self.probe_radius = probe_radius
        self.n_points = n_points

        self.radii_dict = dict()
        self.radii_dict.update(vdwradii)

        self.radii = np.vectorize(self.radii_dict.get)(self.ag.elements)
        self.radii += self.probe_radius
        self.max_radii = 2 * np.max(self.radii)

        self.sphere = self.get_sphere()

    def get_sphere(self) -> np.ndarray:
        """
        Returns a fibonacci unit sphere for a given number of points.

        Returns:
            (np.ndarray): Array of points in fibonacci sphere.
        """
        dl = np.pi * (3 - np.sqrt(5))
        dz = 2. / self.n_points
        longitude = 0
        z = 1 - dz / 2

        xyz = np.zeros((self.n_points, 3), dtype=np.float32)
        for i in range(self.n_points):
            r = np.sqrt(1 - z**2)
            xyz[i, :] = [np.cos(longitude) * r, np.sin(longitude) * r, z]

            z -= dz
            longitude += dl

        return xyz

    def measure_sasa(self,
                     ag: mda.AtomGroup) -> float:
        """
        Measures the SASA of input atomgroup.

        Arguments:
            ag (mda.AtomGroup): MDAnalysis AtomGroup.

        Returns:
            (float): SASA value.
        """
        kdt = KDTree(ag.positions, 10)

        points = np.zeros(ag.n_atoms)
        for i in range(ag.n_atoms):
            sphere = self.sphere.copy() * self.radii[i]
            sphere += ag.positions[i]
            available = self.points_available.copy()
            kdt_sphere = KDTree(sphere, 10)

            for j in kdt.query_ball_point(ag.positions[i], 
                                          self.max_radii, 
                                          workers=-1):
                if j == i:
                    continue
                if self.radii[j] < (self.radii[i] + self.radii[j]):
                    available -= {
                        n for n in kdt_sphere.query_ball_point(
                            self.ag.positions[j],
                            self.radii[j]
                        )
                    }

            points[i] = len(available)

        return 4 * np.pi * self.radii**2 * points / self.n_points

    def _prepare(self):
        """
        Preprocessing step that is run before analyzing the trajectory.
        Initializes the results array.

        Returns:
            None
        """
        self.results.sasa = np.zeros(self.ag.n_residues)

        self.points_available = set(range(self.n_points))

    def _single_frame(self):
        """
        Operations called on each frame. Measures SASA for each atom and sums
        them per-residue, placing the results into internal output array.

        Returns:
            None
        """
        area = self.measure_sasa(self.ag)
        result = np.zeros(self.ag.n_residues)
        for i, atom in enumerate(self.ag.atoms):
            result[atom.resid - 1] += area[i]
        
        self.results.sasa += result

    def _conclude(self):
        """
        Post-processing step to average the SASA per-frame.

        Returns:
            None
        """
        if self.n_frames != 0:
            self.results.sasa /= self.n_frames
            
class RelativeSASA(SASA):
    """
    Computese Relative SASA for a specified atomgroup. Relative SASA is defined
    by the measured SASA divided by the maximum accessible surface area and for
    proteins we define this as within a tripeptide context so as to not overestimate
    SASA of the amide linkage and its neighbors.

    Arguments:
        ag (mda.AtomGroup): AtomGroup for which to perform SASA.
        probe_radius (float): Defaults to 1.4Å. This is pretty standard and should not
            be altered in most cases.
        n_points (int): Defaults to 256. Number of points to place on each sphere, where
            a higher number is more costly but accurate.
    """
    def __init__(self, 
                 ag: mda.AtomGroup,
                 probe_radius: float=1.4,
                 n_points: int=256,
                 **kwargs):
        if not hasattr(ag, 'bonds'):
            raise ValueError('Universe has no `bonds` property!')
        super(RelativeSASA, self).__init__(ag, probe_radius, n_points, **kwargs)

    def _prepare(self):
        """
        Preprocessing step that is run before analyzing the trajectory.
        Initializes the results array.

        Returns:
            None
        """
        self.results.sasa = np.zeros(self.ag.n_residues)
        self.results.relative_area = np.zeros(self.ag.n_residues)

        self.points_available = set(range(self.n_points))

    def _single_frame(self):
        """
        Operations called on each frame. Measures SASA for each atom and sums
        them per-residue. Then computes the exposed area of each tripeptide 
        and stores the ratio of SASA / exposed area in output array.

        Returns:
            None
        """
        area = self.measure_sasa(self.ag)
        result = np.zeros(self.ag.n_residues)
        for i, atom in enumerate(self.ag.atoms):
            result[atom.resid - 1] += area[i]
        
        self.results.sasa += result

        for res_index in self.ag.residues.resindices:
            tri_peptide = self.ag.select_atoms(f'byres (bonded resindex {res_index})')

            if len(tri_peptide) == 0:
                continue

            tri_pep_area = self.measure_sasa(tri_peptide)
            exposed_area = sum([
                a for a, _id in zip(tri_pep_area, tri_peptide.resindices)
                if _id == res_index
            ])

            if exposed_area != 0.:
                result[resindex] /= exposed_area

        self.results.relative_area += np.array([result[_id] for _id in self.ag.residues.resids])

    def _conclude(self):
        """
        Post-processing step to average the SASA per-frame.

        Returns:
            None
        """
        if self.n_frames != 0:
            self.results.sasa /= self.n_frames
            self.results.relative_area /= self.n_frames
