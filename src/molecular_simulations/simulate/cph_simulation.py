from datetime import datetime
import json
import logging
from openmm.app import (CutoffNonPeriodic, 
                        HBonds,
                        ForceField, 
                        LangevinIntegrator, 
                        PDBFile, 
                        PME,
                        Topology)
from openmm.unit import kelvin, nanometers, picosecond
import MDAnalysis as mda
from .constantph.constantph import ConstantPH
from .constantph.logging import setup_task_logger
import parsl
from parsl import Config, python_app
from pathlib import Path
from typing import Any, Optional
import uuid

@python_app
def run_cph_sim(params: dict[str, Any],
                temperature: kelvin,
                n_cycles: int,
                n_steps: int,
                log_params: dict[str, str],
                path: str) -> None:
    variants = params['residueVariants']

    logger = setup_task_logger(**log_params)
    
    cph = ConstantPH(**params)
    cph.simulation.minimizeEnergy()

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
    log_path = log_dir / f'{worker_id}.jsonl'
    
    logger = logging.getLogger(f'task.{task_id}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path)
    handler.setFormatter(JsonFormatter(task_id=worker_id))
    logger.addHandler(handler)

    return logger

class ConstantPHEnsemble:
    def __init__(self,
                 path: list[Path],
                 reference_energies: dict[str, list[float]],
                 parsl_config: Config,
                 log_dir: Path,
                 pHs: list[float]=[x+0.5 for x in range(14)],
                 variant_sel: Optional[str]=None):
        self.paths = paths
        self.ref_energies = reference_energies
        
        self.parsl_config = parsl_config
        self.dfk = None

        self.log_dir = log_dir
        self.pHs = pHs
        self.variant_sel = variant_sel

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def initialize(self) -> None:
        self.dfk = parsl.load(self.parsl_config)

    def shutdown(self) -> None:
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    def load_files(self,
                   path: Path) -> tuple[Topology, np.ndarray]:
        pdb = PDBFile(str(path / 'system.pdb'))
        
        return pdb.topology, pdb.positions

    def build_dicts(self,
                    top: Topology) -> tuple[dict[str, list[str]], 
                                            dict[int, list[float]]]:
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
        
        u = mda.Universe(str(path / 'system.pdb'))
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
        temperature = 300*kelvin
        futures = []
        for i, path in enumerate(self.paths):
            top, pos = self.load_files(path)
            cph_params = self.get_params()
            variants, reference_energies = self.build_dicts(top)

            cph_params.update({
                'topology': top,
                'positions': pos,
                'residueVariants': variants, 
                'referenceEnergies': reference_energies,
            })

            log_params = {
                'run_id': self.run_id,
                'task_id': f'{i:05d}',
                'log_dir': self.log_dir,
            }
    
            futures.append(
                run_cph_sim(cph_params, temperature, n_cycles, n_steps, log_params, str(path))
            )

        _ = [x.result() for x in futures]

    
    @property
    def params(self) -> dict[str, Any]: 
        expl_ff = ForceField('amber14-all.xml', 'amber14/opc.xml')
        impl_ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        expl_params = dict(nonbondedMethod=PME, 
                           nonbondedCutoff=0.9*nanometers, 
                           constraints=HBonds, 
                           hydrogenMass=1.5*amu)
    
        impl_params = dict(nonbondedMethod=CutoffNonPeriodic, 
                           nonbondedCutoff=2.0*nanometers, 
                           constraints=HBonds)
    
        integrator = LangevinIntegrator(temperature, 
                                        1.0/picosecond, 
                                        0.004*picosecond)
        relaxation_integrator = LangevinIntegrator(temperature, 
                                                   10.0/picosecond, 
                                                   0.002*picosecond)
    
        params = {
            'pH': self.pHs,
            'explicitForceField': expl_ff, 
            'implicitForceField': impl_ff, 
            'relaxationSteps': 1000,
            'explicitArgs': expl_p, 
            'implicitArgs': impl_p, 
            'integrator': integrator, 
            'relaxationIntegrator': relax_integrator,
        }

        return params
