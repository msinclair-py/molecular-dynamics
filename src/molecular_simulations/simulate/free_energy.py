"""
"""
from copy import deepcopy
from openmm import CustomBondForce, CustomCompoundBondForce, HarmonicBondForce
from openmm.unit import angstrom
import numpy as np
import parsl
from parsl import python_app, Config
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10
import traceback
from typing import Any, Optional, Type, TypeVar
from omm_simulator import ImplicitSimulator, Simulator
from reporters import RCReporter

_T = TypeVar('_T')
    
@python_app
def run_evb_window(topology: Path,
                   coord_file: Path,
                   out_path: Path,
                   rc_file: Path,
                   umbrella_force: dict[str, int | float],
                   morse_bond: dict[str, int | float],
                   rc_freq: int,
                   steps: int,
                   dt: float,
                   platform: str,
                   restraint_sel: str | None) -> None:
    """Parsl python app. Separate module due to need for serialization.
    """
    evb = EVBCalculation(
        topology=topology,
        coord_file=coord_file,
        out_path=out_path,
        rc_file=rc_file,
        umbrella=umbrella_force,
        morse_bond=morse_bond,
        rc_freq=rc_freq,
        steps=steps,
        dt=dt,
        platform=platform,
        restraint_sel=restraint_sel
    )

    evb.run()

class EVB:
    """EVB orchestrator. Sets up full EVB run for a set of reactants or products,
    and distributes calculations using Parsl."""
    def __init__(self,
                 topology: Path,
                 coordinates: Path,
                 umbrella_atoms: list[int],
                 morse_atoms: list[int],
                 reaction_coordinate:  list[float],
                 parsl_config: Config,
                 log_path: Path,
                 log_prefix: str='reactant',
                 rc_write_freq: int=5,
                 steps: int=500000,
                 dt: float=0.002,
                 k: float=160000.0,        # Umbrella force constant (kJ/mol/nm^2)
                 k_path: float=100.0,      # Path restraint force constant (kJ/mol)
                 D_e: float=392.46,        # Morse well depth (kJ/mol) - from BDE
                 alpha: float=13.275,      # Morse width parameter (nm^-1) - computed from sqrt(k_bond/(2*D_e))
                 r0: float=0.1,            # Equilibrium bond distance (nm)
                 platform: str='CUDA',
                 restraint_sel: Optional[str]=None):
        self.topology = Path(topology)
        self.coordinates = Path(coordinates)
        self.umbrella_atoms = umbrella_atoms
        self.morse_atoms = morse_atoms
        self.path = self.topology.parent / 'evb'

        self.parsl_config = parsl_config
        self.dfk = None

        self.log_path = Path(log_path)
        self.log_prefix = log_prefix
        self.rc_freq = rc_write_freq

        self.steps = steps
        self.dt = dt
        self.k = k
        self.k_path = k_path
        self.D_e = D_e
        self.alpha = alpha
        self.r0 = r0
        
        self.platform = platform
        self.restraint_sel = restraint_sel

        self.reaction_coordinate = self.construct_rc(reaction_coordinate)

        self.inspect_inputs()

    def inspect_inputs(self) -> None:
        """Inspects EVB inputs to assure everything is formatted correctly.
        
        Raises:
            AssertionError: Any input file is missing or the indices are set
                wrong. Additionally checks to make sure reaction coordinate
                is set correctly, having more than 1 window.
        """
        paths = [
            self.topology, self.coordinates,
        ]

        for path in paths:
            assert path.exists(), f'File {path} not found!'

        assert len(self.umbrella_atoms) == 3, f'Need 3 umbrella atoms!'
        assert len(self.morse_atoms) == 2, f'Need 2 morse bond atoms!'
        assert self.reaction_coordinate.shape[0] > 1, f'RC needs at least 1 window!'

        self.log_path.mkdir(exist_ok=True, parents=True)

    def construct_rc(self,
                     rc: list[float]) -> np.ndarray:
        """Construct linearly spaced reaction coordinate.

        Args:
            rc (tuple[float]): (rc_minimum, rc_maximum, rc_increment)

        Returns:
            (np.ndarray): Linearly spaced reaction coordinate
        """
        return np.arange(rc[0], rc[1] + rc[2], rc[2])


    def initialize(self) -> None:
        """Initialize Parsl for runs"""
        if self.dfk is None:
            self.dfk = parsl.load(self.parsl_config)

    def shutdown(self) -> None:
        """Clean up Parsl after runs"""
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None

        parsl.clear()

    def run_evb(self) -> None:
        """Collect futures for each EVB window and distribute."""
        # Spin up Parsl
        self.initialize()

        try:
            futures = []
            for i, rc0 in enumerate(self.reaction_coordinate):
                umbrella = {**self.umbrella, 'rc0': rc0}
                
                futures.append(
                    run_evb_window(
                        topology=self.topology,
                        coord_file=self.coordinates,
                        out_path=self.path / f'window{i}',
                        rc_file=self.log_path / f'{self.log_prefix}_{i}.log',
                        umbrella_force=umbrella,
                        morse_bond=self.morse_bond,
                        rc_freq=self.rc_freq,
                        steps=self.steps,
                        dt=self.dt,
                        platform=self.platform,
                        restraint_sel=self.restraint_sel,
                    )
                )

            _ = [x.result() for x in futures]

        except Exception as e:
            tb = traceback.format_exc()
            print(
                'EVB failed for 1 or more windows!'
                f'{e}'
                f'{tb}'
            )

        finally:
            # Stop Parsl to avoid zombie processes
            self.shutdown()
    
    @property
    def umbrella(self) -> dict[str, Any]:
        """Sets up Umbrella force settings for force calculation.

        Because the windows are decided at run time we leave rc0 as None for now.
        The dict includes both k (umbrella force constant) and k_path (path
        restraint force constant) since both are passed to the EVBCalculation.

        Returns:
            dict: Umbrella and path restraint parameters.
        """
        return {
            'atom_i': self.umbrella_atoms[0],
            'atom_j': self.umbrella_atoms[1],
            'atom_k': self.umbrella_atoms[2],
            'k': self.k,
            'k_path': self.k_path,
            'rc0': None
        }

    @property
    def morse_bond(self) -> dict[str, Any]:
        """Sets up Morse bond settings for potential creation.

        D_e is the potential well depth (bond dissociation energy) and can be
        computed using QM or obtained from ML predictions such as ALFABET
        (https://bde.ml.nrel.gov). Must be in kJ/mol for OpenMM.

        alpha is the potential well width computed from the harmonic force
        constant via the Taylor expansion of the second derivative:
            alpha = sqrt(k_bond / (2 * D_e))

        Unit conversions from AMBER frcmod (kcal/mol/A^2) to OpenMM (kJ/mol/nm^2):
            k_openmm = k_amber * 4.184 * 100

        Example calculation for C-H bond:
            D_e = 93.8 kcal/mol = 392.46 kJ/mol
            k_bond = 330.6 kcal/(mol*A^2) = 138323 kJ/(mol*nm^2)
            alpha = sqrt(138323 / (2 * 392.46)) = 13.275 nm^-1

        r0 is the equilibrium bond distance in nm.

        Returns:
            dict: Morse bond parameters with keys atom_i, atom_j, D_e, alpha, r0.
        """
        return {
            'atom_i': self.morse_atoms[0],
            'atom_j': self.morse_atoms[1],
            'D_e': self.D_e,
            'alpha': self.alpha,
            'r0': self.r0,
        }

    
class EVBCalculation:
    """Runs a single EVB window."""
    def __init__(self,
                 topology: Path,
                 coord_file: Path,
                 out_path: Path,
                 rc_file: Path,
                 umbrella: dict,
                 morse_bond: dict,
                 rc_freq: int=5, # 0.01 ps @ 2 fs timestep 
                 steps: int=500_000, # 1 ns @ 2 fs timestep
                 dt: float=0.002,
                 platform: str='CUDA',
                 restraint_sel: Optional[str]=None):
        self.sim_engine = Simulator(
            path = topology.parent,
            top_name = topology.name,
            coor_name = coord_file.name,
            out_path = out_path,
            prod_steps=steps,
            platform=platform,
        )

        # Only set Precision for platforms that support it (CUDA, OpenCL)
        if platform.upper() in ('CUDA', 'OPENCL'):
            self.sim_engine.properties = {
                'Precision': 'mixed',
            }
        else:
            self.sim_engine.properties = {}

        self.rc_file = rc_file
        self.rc_freq = rc_freq
        self.steps = steps
        self.dt = dt
        self.restraint_sel = restraint_sel
        self.umbrella = umbrella
        self.morse_bond = morse_bond
        
    def prepare(self):
        """Generates simulation object containing all custom forces to compute
        free energy. Leverages standard Simulator as backend, adding in Morse
        potential and Umbrella forces.
        """
        # load files into system object
        system = self.sim_engine.load_system()

        # Remove the original harmonic bond before adding Morse potential
        # to avoid double-counting the bonded interaction
        self.remove_harmonic_bond(
            system,
            self.morse_bond['atom_i'],
            self.morse_bond['atom_j']
        )

        # add various custom forces to system
        morse_bond = self.morse_bond_force(**self.morse_bond)
        system.addForce(morse_bond)
        ddbonds_umb = self.umbrella_force(**self.umbrella)
        system.addForce(ddbonds_umb)
        path_force = self.path_restraint(**self.umbrella)
        system.addForce(path_force)

        # if we want restraints add them now
        if self.restraint_sel is not None:
            restraint_idx = self.sim_engine.get_restraint_indices(self.restraint_sel)
            system = self.sim_engine.add_backbone_posres(
                system,
                self.sim_engine.coordinate.positions,
                self.sim_engine.topology.topology.atoms(),
                restraint_idx,
            )

        # finally, build simulation object
        simulation, integrator = self.sim_engine.setup_sim(system, dt=self.dt)
        simulation.context.setPositions(self.sim_engine.coordinate.positions)
        
        return simulation, integrator

    def run(self):
        """Runs EVB simulation window with custom RCReporter."""
        simulation, integrator = self.prepare()
        simulation.minimizeEnergy()
        simulation = self.sim_engine.attach_reporters(simulation,
                                                      self.sim_engine.dcd,
                                                      str(self.sim_engine.prod_log),
                                                      str(self.sim_engine.restart),
                                                      restart=False)
        atom_indices = [
            self.umbrella['atom_i'],
            self.umbrella['atom_j'],
            self.umbrella['atom_k'],
        ]
        
        simulation.reporters.append(
            RCReporter(self.rc_file, self.rc_freq, atom_indices, self.umbrella['rc0'])
        )

        simulation.step(self.steps)
    
    @staticmethod
    def umbrella_force(atom_i: int,
                       atom_j: int,
                       atom_k: int,
                       k: float,
                       rc0: float,
                       **kwargs) -> CustomCompoundBondForce:
        """Difference of distances umbrella force. Think pulling an oxygen off

        Args:
            atom_i (int): Index of first atom participating (from reactant).
            atom_j (int): Index of second atom participating (from product).
            atom_k (int): Index of shared atom participating in both reactant and product.
            k (float, optional): Harmonic spring constant.
            rc0 (float, optional): Target equilibrium distance for current window.

        Returns:
            CustomBondForce: Force that drives sampling in each umbrella window.
        """
        force = CustomCompoundBondForce(3, '0.5 * k * ((r13 - r23) - rc0) ^ 2; r13=distance(p1, p3); r23=distance(p2, p3);')
        force.addGlobalParameter('k', k)
        force.addGlobalParameter('rc0', rc0)
        force.addBond([atom_i, atom_j, atom_k])
    
        return force

    @staticmethod
    def path_restraint(atom_i: int,
                       atom_j: int,
                       atom_k: int,
                       k_path: float,
                       **kwargs) -> CustomCompoundBondForce:
        """Enforce collinearity of moving atom with respect to the initial
        and final positions. By avoiding a custom angle force we avoid instability
        related to the asymptote at 180 degrees, which is what we are attempting to
        enforce. The cosine of the dot product of the vectors from i -> k and i -> j
        allows the penalty to scale quadratically with deviation, thus keeping the
        mobile atom snapped along the progress coordinate vector.

        Note: k_path has units of kJ/mol (not kJ/mol/nm^2 like the umbrella k)
        since (1 - costheta)^2 is dimensionless. Typical values are 50-200 kJ/mol.

        Args:
            atom_i (int): Index of donor atom.
            atom_j (int): Index of acceptor atom.
            atom_k (int): Index of transferring atom (e.g., hydride).
            k_path (float): Force constant in kJ/mol for collinearity restraint.

        Returns:
            CustomCompoundBondForce: Force enforcing D-H-A collinearity.
        """
        force = CustomCompoundBondForce(3,  (
                'k_path * (1 - costheta)^2; '
                'costheta = dot_ij_ik / (r_ij * r_ik); '
                'dot_ij_ik = dx_ij*dx_ik + dy_ij*dy_ik + dz_ij*dz_ik; '
                'r_ij = sqrt(dx_ij^2 + dy_ij^2 + dz_ij^2); '
                'r_ik = sqrt(dx_ik^2 + dy_ik^2 + dz_ik^2); '
                'dx_ij = x2 - x1; '
                'dy_ij = y2 - y1; '
                'dz_ij = z2 - z1; '
                'dx_ik = x3 - x1; '
                'dy_ik = y3 - y1; '
                'dz_ik = z3 - z1'
            )
        )
        force.addGlobalParameter('k_path', k_path)
        force.addBond([atom_i, atom_k, atom_j])

        return force
    
    @staticmethod
    def morse_bond_force(atom_i: int,
                         atom_j: int,
                         D_e: float,
                         alpha: float,
                         r0: float) -> CustomBondForce:
        """Generates a custom Morse potential between two atom indices.

        The Morse potential has the form:
            V(r) = D_e * (1 - exp(-alpha * (r - r0)))^2

        The alpha parameter can be computed from harmonic force constant k_bond:
            alpha = sqrt(k_bond / (2 * D_e))

        All parameters must be in OpenMM's native unit system:
            - Distances in nanometers (nm)
            - Energies in kJ/mol
            - Force constants in kJ/(mol*nm^2)
            - Alpha in nm^-1

        Args:
            atom_i (int): Index of first atom.
            atom_j (int): Index of second atom.
            D_e (float): Well depth in kJ/mol (from bond dissociation energy).
            alpha (float): Width parameter in nm^-1.
            r0 (float): Equilibrium distance in nm.

        Returns:
            CustomBondForce: Force corresponding to a Morse potential.
        """
        force = CustomBondForce('D_e * (1 - exp(-alpha * (r-r0))) ^ 2')
        force.addGlobalParameter('D_e', D_e)
        force.addGlobalParameter('alpha', alpha)
        force.addGlobalParameter('r0', r0)
        force.addBond(atom_i, atom_j)

        return force

    @staticmethod
    def remove_harmonic_bond(system, atom_i: int, atom_j: int) -> None:
        """Remove the bond/constraint between two atoms.

        This is necessary when replacing a harmonic bond with a Morse potential
        to avoid double-counting the bonded interaction. The method handles both:
        1. Harmonic bonds (sets force constant to zero)
        2. SHAKE/RATTLE constraints (removes the constraint entirely)

        Args:
            system: OpenMM System object containing forces.
            atom_i (int): Index of first atom in the bond.
            atom_j (int): Index of second atom in the bond.

        Returns:
            None. Modifies system in place.
        """
        target_pair = {atom_i, atom_j}
        found_bond = False
        found_constraint = False

        # First, check for harmonic bond and zero it out
        for force_idx in range(system.getNumForces()):
            force = system.getForce(force_idx)
            if isinstance(force, HarmonicBondForce):
                for bond_idx in range(force.getNumBonds()):
                    p1, p2, length, k = force.getBondParameters(bond_idx)
                    if {p1, p2} == target_pair:
                        # Zero out the force constant, keeping equilibrium length
                        force.setBondParameters(bond_idx, p1, p2, length, 0.0)
                        print(f"Zeroed harmonic bond between atoms {atom_i} and {atom_j}")
                        found_bond = True
                        break
                break

        # Second, check for constraints (SHAKE) and remove them
        # Need to iterate in reverse since we're removing
        constraints_to_remove = []
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if {p1, p2} == target_pair:
                constraints_to_remove.append(i)

        # Remove constraints in reverse order to maintain indices
        for idx in reversed(constraints_to_remove):
            system.removeConstraint(idx)
            print(f"Removed SHAKE constraint between atoms {atom_i} and {atom_j}")
            found_constraint = True

        if not found_bond and not found_constraint:
            print(f"Warning: No harmonic bond or constraint found between atoms {atom_i} and {atom_j}")

