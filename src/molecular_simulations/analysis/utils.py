import MDAnalysis as mda
import numpy as np
from pathlib import Path
import shutil
from typing import Callable, Union

OptPath = Union[Path, str, None]

class EmbedData:
    """
    Embeds given data into the beta-factor column of PDB. Writes out to same
    path as input PDB and backs up old PDB file, unless an output path is 
    explicitly provided. Embedding data should be provided as a dictionary where
    the keys are MDAnalysis selection strings and the values are numpy arrays
    of shape (n_frames, n_residues, n_datapoints) or (n_residues, n_datapoints).
    """
    def __init__(self,
                 pdb: Path,
                 embedding_dict: dict[str, np.ndarray],
                 out: OptPath=None):
        self.pdb = pdb if isinstance(pdb, Path) else Path(pdb)
        self.embeddings = embedding_dict
        self.out = out if out is not None else self.pdb
        
        self.u = mda.Universe(str(self.pdb))

    def embed(self):
        for sel, data in self.embeddings.items():
            self.embed_selection(sel, data)

        self.write_new_pdb()

    def embed_selection(self,
                        selection: str,
                        data: np.ndarray) -> None:
        sel = self.u.select_atoms(selection)

        for residue, datum in zip(sel.residues, data):
            residue.atoms.tempfactors = np.full(residue.atoms.tempfactors.shape, datum)
    
    def write_new_pdb(self) -> None:
        if self.out.exists():
            if not self.pdb.with_suffix('.orig.pdb').exists():
                shutil.copyfile(str(self.pdb), str(self.pdb.with_suffix('.orig.pdb')))

        with mda.Writer(str(self.out)) as W:
            W.write(self.u.atoms)


class EmbedEnergyData(EmbedData):
    """

    """
    def __init__(self,
                 pdb: Path,
                 embedding_dict: dict[str, np.ndarray],
                 out: OptPath=None):
        super().__init__(pdb, embedding_dict, out)
        self.embeddings = self.preprocess()

    def preprocess(self) -> dict[str, np.ndarray]:
        new_embeddings = dict()
        all_data = []
        for sel, data in self.embeddings.items():
            sanitized = self.sanitize_data(data)
            all_data.append(sanitized)

        rescaling_factor = np.min(np.concatenate(all_data))
        for sel, data in self.embeddings.items():
            sanitized = self.sanitize_data(data)
            rescaled = sanitized / rescaling_factor
            rescaled[np.where(rescaled > 1.)] = 1.
            new_embeddings[sel] = rescaled

        return new_embeddings

    @staticmethod
    def sanitize_data(data: np.ndarray,) -> np.ndarray:
        if len(data.shape) > 2:
            data = np.mean(data, axis=0)

        if data.shape[1] > 1:
            data = np.sum(data, axis=1)
        
        return data
