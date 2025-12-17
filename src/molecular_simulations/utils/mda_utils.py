import MDAnalysis as mda
import numpy as np
from pathlib import Path
from rust_simulation_tools import (kabsch_align,
                                   unwrap_system,
                                   rewrap_system)
from typing import Optional

def trim_trajectory(u: mda.Universe,
                    out: Path,
                    stride: int=1,
                    align: bool=False,
                    rewrap: bool=False,
                    sel: Optional[str]=None,
                    align_sel: Optional[str]=None,) -> None:
    if sel is not None:
        selection = u.select_atoms(sel)
    else:
        selection = u.atoms

    positions = np.zeros((u.trajectory.n_frames // stride, selection.n_atoms, 3), dtype=np.float32)

    for i, ts in u.trajectory[::stride]:
        positions[i, ...] = selection.positions.copy().astype(np.float32)

    if align:
        if align_sel is None:
            align_idx = selection.select_atoms('backbone or nucleicbackbone').ix
        else:
            align_idx = selection.select_atoms(align_sel).ix

        positions = kabsch_align(positions, positions[0], align_idx)

    if rewrap:
        pass

    with mda.Writer(str(out), n_atoms=selection.n_atoms) as W:
        for pos in positions:
            W.write(pos)
