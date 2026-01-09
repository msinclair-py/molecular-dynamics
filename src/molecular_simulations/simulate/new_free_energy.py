"""
"""
from copy import deepcopy
from openmm import CustomBondForce, CustomCompoundBondForce
from openmm.unit import angstrom
import numpy as np
import parsl
from parsl import python_app, Config
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10
from typing import Optional, Type, TypeVar
from .omm_simulator import ImplicitSimulator, Simulator
from .reporters import RCReporter

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

    Args:
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
    """EVB orchestrator. Sets up full EVB run for a set of reactants and products,
    and distributes calculations using Parsl."""
    def __init__(self,
                 topology: Path,
                 coordinates: Path,
                 umbrella_atoms: list[int],
                 morse_atoms: list[int],
                 reaction_coordinate:  list[float],
                 log_path: Path,
                 log_prefix: str,
                 parsl_config: Config,
                 rc_write_freq: int=5,
                 steps: int=500000,
                 dt: float=0.002,
                 k: float=160000.0,
                 D_e: float=1491.89,
                 alpha: float=10.46,
                 r0: float=0.1,
                 platform: str='CUDA',
                 restraint_sel: Optional[str]=None):
        self.topology = Path(topology)
        self.coordinates = Path(coordinates)
        self.umbrella_atoms = umbrella_atoms
        self.morse_atoms = morse_atoms
        self.path = self.topology.parent / 'evb'
        
        self.log_path = Path(log_path)
        self.log_prefix = log_prefix
        self.rc_freq = rc_write_freq

        self.steps = steps
        self.dt = dt
        self.k = k
        self.D_e = D_e
        self.alpha = alpha
        self.r0 = r0
        
        self.platform = platform
        self.restraint_sel = restraint_sel

        self.reaction_coordinate = self.construct_rc(reaction_coordinate)
        self.parsl_config = parsl_config
        self.dfk = None

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
        futures = []
        for i, rc0 in enumerate(self.reaction_coordinate):
            umbrella = self.umbrella.update({'rc0': rc0})
            
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
    
    @property
    def umbrella(self,
                 k:  float) -> dict[str, Any]:
        """Sets up Umbrella force settings for force calculation. Because the
        windows are decided at run time we leave rc0 as None for now.

        Args:
            k (float): pass

        Returns:
            dict
        """
        return {
            'atom_i': self.umbrella_atoms[0],
            'atom_j': self.umbrella_atoms[1],
            'atom_k': self.umbrella_atoms[2],
            'k': k,
            'rc0': None
        }

    @property
    def morse_bond(self) -> dict[str, Any]:
        """Sets up Morse bond settings for potential creation. D_e is the 
        potential well depth and can be computed using QM or scraped from
        ML predictions such as ALFABET (https://bde.ml.nrel.gov). alpha is 
        the potential well width and is computed from the Taylor expansion of
        the second derivative of the potential. This takes the form:
        .. math::
            \alpha = \sqrt(\frac{k_e}{2*D_e})

        Finally, r0 is the equilibrium bond distance coming from literature,
        forcefield, QM or ML predictions.

        Args:
            None

        Returns:
            None
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

        self.sim_engine.properties = {
            'Precision': 'mixed',
        }

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
                       rc0: float) -> CustomCompoundBondForce:
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
                       k: float,
                       **kwargs) -> CustomCompoundBondForce:
        """Enforce collinearity of moving atom with respect to the initial
        and final positions. By avoiding a custom angle force we avoid instability
        related to the asymptote at 180 degrees, which is what we are attempting to
        enforce. The cosine of the dot product of the vectors from i -> k and i -> j
        allows the penalty to scale quadratically with deviation, thus keeping the 
        mobile atom snapped along the progress coordinate vector."""
        force = CustomCompoundBondForce(3,  (
                'k * (1 - costheta)^2; '
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
        force.addGlobalParameter('k', k)
        force.addBond([atom_i, atom_k, atom_j])

        return force
    
    @staticmethod
    def morse_bond_force(atom_i: int,
                         atom_j: int,
                         D_e: float,
                         alpha: float,
                         r0: float) -> CustomBondForce:
        """Generates a custom Morse potential between two atom indices.

        Args:
            atom_i (int): Index of first atom
            atom_j (int): Index of second atom
            D_e (float, optional): Depth of the Morse potential.
            alpha (float, optional): Stiffness of the Morse potential.
            r0 (float, optional): Equilibrium distance of the bond represented.

        Returns:
            CustomBondForce: Force corresponding to a Morse potential.
        """
        force = CustomBondForce('D_e * (1 - exp(-alpha * (r-r0))) ^ 2')
        force.addGlobalParameter('D_e', D_e)
        force.addGlobalParameter('alpha', alpha)
        force.addGlobalParameter('r0', r0)
        force.addBond(atom_i, atom_j)
        
        return force

