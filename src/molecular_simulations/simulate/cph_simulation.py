"""Constant pH molecular dynamics simulations using OpenMM.

This module provides classes and functions for running constant pH simulations
with replica exchange across different pH values. It uses Parsl for distributed
execution of simulation replicas.
"""

from datetime import datetime
import json
import logging
import MDAnalysis as mda
import numpy as np
from openmm import LangevinIntegrator
from openmm.app import (CutoffNonPeriodic, 
                        HBonds,
                        ForceField, 
                        PDBFile, 
                        PME,
                        Topology)
from openmm.unit import (amu,
                         kelvin, 
                         kilojoules_per_mole,
                         nanometers, 
                         picosecond)
import parsl
from parsl import Config, python_app
from pathlib import Path
from typing import Any, Optional
import uuid
from .constantph.constantph import ConstantPH
from .constantph.logging import setup_task_logger

@python_app
def run_cph_sim(params: dict[str, Any],
                temperature: kelvin,
                n_cycles: int,
                n_steps: int,
                log_params: dict[str, str],
                path: str) -> None:
    """Run a constant pH simulation as a Parsl python app.

    This function executes a single constant pH simulation replica,
    performing Monte Carlo titration steps between MD propagation cycles.

    Args:
        params: Dictionary of ConstantPH parameters including topology,
            coordinates, residue variants, and reference energies.
        temperature: Simulation temperature with OpenMM units.
        n_cycles: Number of MD/MC cycles to perform.
        n_steps: Number of MD steps per cycle.
        log_params: Dictionary containing logging configuration with
            run_id, task_id, and log_dir.
        path: Path to the simulation directory for logging purposes.
    """
    variants = params['residueVariants']

    logger = setup_task_logger(**log_params)
    
    cph = ConstantPH(**params)
    cph.simulation.minimizeEnergy()
    cph.simulation.context.setVelocitiesToTemperature(temperature)

    resids = list(variants.keys())
    
    logger.info(f'Running {n_cycles} cycles of constant pH simulation!',
                extra={'path': path, 'pH': None, 'step': None, 'resids': resids})
    for i in range(n_cycles):
        cph.simulation.step(n_steps)
        cph.attemptMCStep(temperature)
        pH = cph.pH[cph.currentPHIndex]
        current_variants = [variants[index][cph.titrations[index].currentIndex] 
                            for index in variants]
        logger.info(f'Step complete {i}!',
                    extra={'path': path, 'pH': pH, 'step': i, 'resnames': current_variants})

def setup_worker_logger(self,
                        worker_id: str,
                        log_dir: Path) -> logging.Logger:
    """Set up a JSON logger for a simulation worker.

    Creates a logger that writes JSON-formatted log entries to a
    worker-specific file.

    Args:
        worker_id: Unique identifier for the worker.
        log_dir: Directory path where log files will be written.

    Returns:
        Configured logging.Logger instance for the worker.
    """
    log_path = log_dir / f'{worker_id}.jsonl'
    
    logger = logging.getLogger(f'task.{task_id}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path)
    handler.setFormatter(JsonFormatter(task_id=worker_id))
    logger.addHandler(handler)

    return logger

class ConstantPHEnsemble:
    """Orchestrator for running constant pH simulation ensembles.

    Manages distributed execution of multiple constant pH simulation replicas
    using Parsl. Each replica can sample across a range of pH values with
    Monte Carlo titration moves.
    """

    def __init__(self,
                 paths: list[Path],
                 reference_energies: dict[str, list[float]],
                 parsl_config: Config,
                 log_dir: Path,
                 pHs: list[float]=[x+0.5 for x in range(14)],
                 temperature: float=300.,
                 variant_sel: Optional[str]=None,):
        """Initialize the constant pH ensemble.

        Args:
            paths: List of paths to simulation directories, each containing
                system.prmtop and system.inpcrd files.
            reference_energies: Dictionary mapping residue names to lists of
                reference free energies for each protonation state.
            parsl_config: Parsl configuration for distributed execution.
            log_dir: Directory path for writing simulation logs.
            pHs: List of pH values to sample. Defaults to [0.5, 1.5, ..., 13.5].
            temperature: Simulation temperature in Kelvin. Defaults to 300.
            variant_sel: Optional MDAnalysis selection string to limit which
                titratable residues are included. Defaults to None.
        """
        self.paths = paths
        self.ref_energies = reference_energies
        
        self.parsl_config = parsl_config
        self.dfk = None

        self.log_dir = log_dir
        self.pHs = pHs
        self.variant_sel = variant_sel
        
        self.temperature = temperature * kelvin

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def initialize(self) -> None:
        """Initialize Parsl for distributed simulation execution."""
        self.dfk = parsl.load(self.parsl_config)

    def shutdown(self) -> None:
        """Clean up Parsl resources after simulation completion."""
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    def load_files(self,
                   path: Path) -> tuple[Topology, np.ndarray]:
        """Load topology and coordinates from AMBER files.

        Args:
            path: Directory containing system.prmtop and system.inpcrd files.

        Returns:
            Tuple of (OpenMM Topology, atomic positions array).
        """
        prmtop = AmberPrmtopFile(str(path / 'system.prmtop'))
        inpcrd = AmberInpcrdFile(str(path / 'system.inpcrd'))

        return prmtop.topology, inpcrd.positions

    def build_dicts(self,
                    path: Path,
                    top: Topology) -> tuple[dict[str, list[str]],
                                            dict[int, list[float]]]:
        """Build residue variant and reference energy dictionaries.

        Identifies titratable residues in the topology and maps them to
        their possible protonation states and corresponding reference energies.

        Args:
            path: Path to the structure file for MDAnalysis loading.
            top: OpenMM Topology object containing residue information.

        Returns:
            Tuple of (variants dict, reference_energies dict) where variants
            maps residue indices to lists of residue name variants and
            reference_energies maps residue indices to lists of reference
            free energies in kJ/mol.
        """
        _variants = {
            'CYS': ['CYS', 'CYX'],
            'ASP': ['ASH', 'ASP'],
            'GLU': ['GLH', 'GLU'],
            'LYS': ['LYS', 'LYN'],
            'HIS': ['HIP', 'HID', 'HIE'],
        }
        names = list(_variants.keys())
        variants = {}
        reference_energies = {}
        for residue in top.residues():
            if residue.name in names:
                variants[residue.index] = _variants[residue.name]
                reference_energies[residue.index] = [x * kilojoules_per_mole 
                                                     for x in self.ref_energies[residue.name]]
        
        u = mda.Universe(str(path))
        sel = u.select_atoms('protein')
        bad_keys = [sel[0].resid, sel[-1].resid] # termini
        
        if self.variant_sel is not None:
            var = sel.select_atoms(self.variant_sel)
            bad_keys = [resid 
                        for resid in sel.residues.resids 
                        if resid not in var.residues.resids]

        for bad_key in bad_keys:
            if bad_key in variants:
                del variants[bad_key]
            if bad_key in reference_energies:
                del reference_energies[bad_key]
    
        return variants, reference_energies
    
    def run(self,
            n_cycles: int=500,
            n_steps: int=500) -> None:
        """Run the constant pH simulation ensemble.

        Distributes simulation replicas across available resources using
        Parsl and waits for all replicas to complete.

        Args:
            n_cycles: Number of MD/MC cycles per replica. Defaults to 500.
            n_steps: Number of MD steps per cycle. Defaults to 500.
        """
        futures = []
        for i, path in enumerate(self.paths):
            top, pos = self.load_files(path)
            variants, reference_energies = self.build_dicts(path, top)

            cph_params = self.params

            cph_params.update({
                'residueVariants': variants, 
                'referenceEnergies': reference_energies,
            })

            log_params = {
                'run_id': self.run_id,
                'task_id': f'{i:05d}',
                'log_dir': self.log_dir,
            }
    
            futures.append(
                run_cph_sim(cph_params, self.temperature, n_cycles, n_steps, log_params, str(path))
            )

        _ = [x.result() for x in futures]

    
    @property
    def params(self) -> dict[str, Any]:
        """Build the parameter dictionary for ConstantPH initialization.

        Returns:
            Dictionary containing all parameters needed to initialize a
            ConstantPH simulation including file paths, pH values, integrators,
            and force field parameters for both explicit and implicit solvent.
        """
        expl_params = dict(nonbondedMethod=PME, 
                           nonbondedCutoff=0.9*nanometers, 
                           constraints=HBonds, 
                           hydrogenMass=1.5*amu)
    
        impl_params = dict(nonbondedMethod=CutoffNonPeriodic, 
                           nonbondedCutoff=2.0*nanometers, 
                           constraints=HBonds)
    
        integrator = LangevinIntegrator(self.temperature, 
                                        1.0/picosecond, 
                                        0.004*picosecond)
        relaxation_integrator = LangevinIntegrator(self.temperature, 
                                                   10.0/picosecond, 
                                                   0.002*picosecond)
    
        params = {
            'prmtop_file': self.path / 'system.prmtop',
            'inpcrd_file': self.path / 'system.inpcrd',
            'pH': self.pHs,
            'relaxationSteps': 1000,
            'explicitArgs': expl_params, 
            'implicitArgs': impl_params, 
            'integrator': integrator, 
            'relaxationIntegrator': relaxation_integrator,
        }

        return params
