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

    Arguments:
        pdb (Path): Path to PDB file to load. Also will be the output if one is
            not provided.
        embedding_dict (dict[str, np.ndarray]): A dictionary containing MDAnalysis
            selections as keys and data as the values.
        out (OptPath): Defaults to None. If not None this will be the path to the
            output PDB.
    """
    def __init__(self,
                 pdb: Path,
                 embedding_dict: dict[str, np.ndarray],
                 out: OptPath=None):
        self.pdb = pdb if isinstance(pdb, Path) else Path(pdb)
        self.embeddings = embedding_dict
        self.out = out if out is not None else self.pdb
        
        self.u = mda.Universe(str(self.pdb))

    def embed(self) -> None:
        """
        Unpacks embedding dictionary, embeds data and writes out new PDB.

        Returns:
            None
        """
        for sel, data in self.embeddings.items():
            self.embed_selection(sel, data)

        self.write_new_pdb()

    def embed_selection(self,
                        selection: str,
                        data: np.ndarray) -> None:
        """
        Embeds data into given selection in the beta column for each residue.

        Arguments:
            selection (str): MDAnalysis selection string.
            data (np.ndarray): Array of data to place in beta column. Shape should
                be (n_residues_in_selection, 1).

        Returns:
            None
        """
        sel = self.u.select_atoms(selection)

        for residue, datum in zip(sel.residues, data):
            residue.atoms.tempfactors = np.full(residue.atoms.tempfactors.shape, datum)
    
    def write_new_pdb(self) -> None:
        """
        Writes out PDB file. If an output was not designated, backs up original
        PDB with the extension .orig.pdb. If this backup already exists, do not
        back up the PDB as that may occur if you run this twice and to do so would
        mean losing the actual original PDB.

        Returns:
            None
        """
        if self.out.exists():
            if not self.pdb.with_suffix('.orig.pdb').exists():
                shutil.copyfile(str(self.pdb), str(self.pdb.with_suffix('.orig.pdb')))

        with mda.Writer(str(self.out)) as W:
            W.write(self.u.atoms)


class EmbedEnergyData(EmbedData):
    """
    Special instance of EmbedData in which the data stored in embedding_dict is
    non-bonded energy data with both LJ and coulombic terms. In this case we need
    to obtain the total energy by summing these and rescale it as many softwares
    do not understand a negative beta factor.

    Arguments:
        pdb (Path): Path to PDB file to load. Also will be the output if one is
            not provided.
        embedding_dict (dict[str, np.ndarray]): A dictionary containing MDAnalysis
            selections as keys and data as the values.
        out (OptPath): Defaults to None. If not None this will be the path to the
            output PDB.
    """
    def __init__(self,
                 pdb: Path,
                 embedding_dict: dict[str, np.ndarray],
                 out: OptPath=None):
        super().__init__(pdb, embedding_dict, out)
        self.embeddings = self.preprocess()

    def preprocess(self) -> dict[str, np.ndarray]:
        """
        Processes embeddings data so that it can be fed through parent methods.
        This requires the embeddings data contain values of one-dimensional arrays,
        and that the data be rescaled such that there are no negative values while
        preserving the distance between values.

        Returns:
            (dict[str, np.ndarray]): Processed data array.
        """
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
        """
        Takes in data of shape (n_frames, n_residues, n_terms) and 
        returns one-dimensional array of shape (n_residues,) by
        first averaging in the first dimension and then summing in
        the new second dimension - originally the third dimension.

        Arguments:
            data (np.ndarray): Unprocessed input data.

        Returns:
            (np.ndarray): One-dimensional processed data.
        """
        if len(data.shape) > 2:
            data = np.mean(data, axis=0)

        if data.shape[1] > 1:
            data = np.sum(data, axis=1)
        
        return data
