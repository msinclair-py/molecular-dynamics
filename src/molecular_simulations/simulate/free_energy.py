"""
"""

from openmm import *
from openmm.app import *
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
    """"""
    def __init__(self,
                 reactant_topology: Path,
                 reactant_coordinates: Path,
                 reactant_umbrella: list[int],
                 reactant_morse: list[int],
                 product_topology: Path,
                 product_coordinates: Path,
                 product_umbrella: list[int],
                 product_morse: list[int],
                 reaction_coordinate:  tuple[float],
                 log_path: Path,
                 parsl_config: Config,
                 rc_write_freq: int=5,
                 steps: int=500000,
                 dt: float=0.002,
                 k: float=160000.0,
                 platform: str='CUDA',
                 restraint_sel: Optional[str]=None,
                 D_e: Optional[float]=None,
                 alpha: Optional[float]=None,
                 r0: Optional[float]=None):
        self.reactant_topology = Path(reactant_topology)
        self.reactant_coordinates = Path(reactant_coordinates)
        self.r_umb = reactant_umbrella
        self.r_mor = reactant_morse
        self.r_path = self.reactant_topology.parent / 'evb'
        
        self.product_topology = Path(product_topology)
        self.product_coordinates = Path(product_coordinates)
        self.p_umb = product_umbrella
        self.p_mor = product_morse
        self.p_path = self.product_topology.parent / 'evb'

        self.log_path = Path(log_path)

        self.rc_freq = rc_write_freq
        self.steps = steps
        self.dt = dt

        self.platform = platform
        self.restraint_sel = restraint_sel

        self.reaction_coordinate = self.construct_rc(reaction_coordinate)
        self.set_umbrella_settings(k)
        self.set_morse_bond_settings(D_e, alpha, r0)
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
            self.reactant_topology, self.reactant_coordinates,
            self.product_topology, self.product_coordinates,
        ]

        for path in paths:
            assert path.exists(), f'File {path} not found!'

        assert len(self.r_umb) == 3, f'Reactant umbrella indices must be length 3!'
        assert len(self.p_umb) == 3, f'Product umbrella indices must be length 3!'
        assert len(self.r_mor) == 2, f'Reactant Morse indices must be length 2!'
        assert len(self.p_mor) == 2, f'Product Morse indices must be length 2!'
        assert self.reaction_coordinate.shape[0] > 1, f'RC needs at least 1 window!'

        self.log_path.mkdir(exist_ok=True, parents=True)

    def construct_rc(self,
                     rc: tuple[float]) -> np.ndarray:
        """Construct linearly spaced reaction coordinate.

        Args:
            rc (tuple[float]): (rc_minimum, rc_maximum, rc_increment)

        Returns:
            (np.ndarray): Linearly spaced reaction coordinate
        """
        return np.arange(rc[0], rc[1] + rc[2], rc[2])

    def set_umbrella_settings(self,
                              k:  float) -> None:
        """Sets up Umbrella force settings for force calculation. Because the
        windows are decided at run time we leave rc0 as None for now.

        Args:
            k (float): pass

        Returns:
            None
        """
        self.r_umbrella = {
            'atom_i': self.r_umb[0],
            'atom_j': self.r_umb[1],
            'atom_k': self.r_umb[2],
            'k': k,
            'rc0': None
        }

        self.p_umbrella = {
            'atom_i': self.p_umb[0],
            'atom_j': self.p_umb[1],
            'atom_k': self.p_umb[2],
            'k': k,
            'rc0': None
        }

    def set_morse_bond_settings(self,
                                D_e: Optional[float],
                                alpha: Optional[float],
                                r0: Optional[float]) -> None:
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
            D_e (Optional[float]): Potential well depth.
            alpha (Optional[float]): Potential width. 
            r0 (Optional[float]): Equilibrium bond distance.
            r_idx (list[int]): List of atom indices from reactant system that
                participate in Morse bond
            p_idx (list[int]): List of atom indices from product system that
                participate in Morse bond

        Returns:
            None
        """
        keys = ['D_e', 'alpha', 'r0']
        settings = {}
        for key, val in zip(keys, [D_e, alpha, r0]):
            if val is not None:
                settings[key] = val
        
        self.r_morse = {
            'atom_i': self.r_mor[0],
            'atom_j': self.r_mor[1],
        }

        self.p_morse = {
            'atom_i': self.p_mor[0],
            'atom_j': self.p_mor[1],
        }

        for morse_dict in [self.r_morse, self.p_morse]:
            morse_dict.update(settings)

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
            self.r_umbrella.update({'rc0': rc0})
            self.p_umbrella.update({'rc0': rc0})

            futures.append(
                run_evb_window(
                    topology=self.reactant_topology,
                    coord_file=self.reactant_coordinates,
                    out_path=self.r_path / f'window{i}',
                    rc_file=self.log_path / f'forward_{i}.log',
                    umbrella_force=self.r_umbrella,
                    morse_bond=self.r_morse,
                    rc_freq=self.rc_freq,
                    steps=self.steps,
                    dt=self.dt,
                    platform=self.platform,
                    restraint_sel=self.restraint_sel,
                )
            )
            
            futures.append(
                run_evb_window(
                    topology=self.product_topology,
                    coord_file=self.product_coordinates,
                    out_path=self.p_path / f'window{i}',
                    rc_file=self.log_path / f'reverse_{i}.log',
                    umbrella_force=self.p_umbrella,
                    morse_bond=self.p_morse,
                    rc_freq=self.rc_freq,
                    steps=self.steps,
                    dt=self.dt,
                    platform=self.platform,
                    restraint_sel=self.restraint_sel,
                )
            )

        _ = [x.result() for x in futures]

    
class EVBCalculation:
    """"""
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

        #self.sim_engine.properties = None

        self.rc_file = rc_file
        self.rc_freq = rc_freq
        self.steps = steps
        self.dt = dt
        self.restraint_sel = restraint_sel
        self.umbrella = umbrella
        self.morse_bond = morse_bond
        
    def prepare(self):
        # load files into system object
        system = self.sim_engine.load_system()
        
        # add various custom forces to system
        morse_bond = self.morse_bond_force(**self.morse_bond)
        system.addForce(morse_bond)
        ddbonds_umb = self.umbrella_force(**self.umbrella)
        system.addForce(ddbonds_umb)

        # if we want restraints add them now
        if self.restraint_sel is not None:
            restraint_idx = self.sim_engine.get_restraint_indices(self.restraint_sel)
            system = self.sim_engine.add_backbone_posres(
                system,
                self.sim_engine.coordinate.positions,
                self.topology.topology.atoms(),
                restraint_idx,
            )

        # finally, build simulation object
        simulation, integrator = self.sim_engine.setup_sim(system, dt=self.dt)
        simulation.context.setPositions(self.sim_engine.coordinate.positions)
        
        return simulation, integrator

    def run(self):
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
                       rc0: float) -> CustomBondForce:
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
        force = CustomCompoundBondForce(3, '0.5 * k * ((r13 - r23) - rc0) ^ 2; r13=periodicdistance(p1, p3); r23=periodicdistance(p2, p3);')
        force.addGlobalParameter('k', k)
        force.addGlobalParameter('rc0', rc0)
        force.addBond([atom_i, atom_j, atom_k])
    
        return force
    
    @staticmethod
    def morse_bond_force(atom_i: int,
                         atom_j: int,
                         D_e: float=419.6, # from NREL server for my O-H
                         alpha: float=2.5977, # a = sqrt(k_OH / (2 * De))
                         r0: float=0.096) -> CustomBondForce:
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

