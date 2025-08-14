import json
import numpy as np
from pathlib import Path
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from typing import Any, Type, TypeVar, Union

PathLike = Union[Path, str]
_T = TypeVar('_T')

class GenericDataloader:
    """Loads any generic data stored in numpy arrays and stores the full
    dataset. Capable of loading data with variable row lengths but must
    be consistent in the columnar dimension.

    Arguments:
        data_files (list[PathLike]): List of paths to input data files.
    """
    def __init__(self,
                 data_files: list[PathLike]):
        self.files = data_files
        self.load_data()

    def load_data(self) -> None:
        """Lumps data into one large array.

        Returns:
            None
        """
        self.data_array = []
        self.shapes = []
        for f in self.files:
            temp = np.load(str(f))
            self.shapes.append(temp.shape)
            self.data_array.append(temp)
        
        self.data_array = np.vstack(self.data_array)
        if len(self.data_array) > 2:
            x, *y = self.data_array.shape
            shape2 = 1
            for shape in y:
                shape2 *= shape

            self.data_array = self.data_array.reshape(x, shape2)

    @property
    def data(self) -> np.ndarray:
        """Property for storing data array.

        Returns:
            (np.ndarray): Internal data array.
        """
        return self.data_array

    @property
    def shape(self) -> tuple[int]:
        """Property for storing shapes of input data.
        
        Returns:
            (tuple[int]): Shape of each individual data file, or if they have 
                different shapes, the shape of each based on the order they were
                provided to this class.
        """
        if len(set(self.shapes)) == 1:
            return self.shapes[0]
        
        return self.shapes


class PeriodicDataloader(GenericDataloader):
    """Decomposes periodic data using sin and cos, returning double the features.
    """
    def __init__(self,
                 data_files: list[PathLike]):
        super().__init__(data_files)
        
    def load_data(self) -> None:
        """Loads file of input periodic data.

        Returns:
            None
        """
        self.data_array = []
        self.shapes = []
        for f in self.files:
            temp = self.remove_periodicity(np.load(str(f)))
            
            self.shapes.append(temp.shape)
            self.data_array.append(temp)

    def remove_periodicity(self,
                           arr: np.ndarray) -> np.ndarray:
        """Removes periodicity from each feature using sin and cos. Each
        column is expanded into two such that the indices become
        i -> 2*i, 2*i + 1.

        Arguments:
            arr (np.ndarray): Data to perform decomposition on.

        Returns:
            (np.ndarray): New array which should be shape (arr.shape[0], arr.shape[1] * 2).
        """
        n_features = arr.shape[1]
        return_arr = np.zeros((arr.shape[0], n_features * 2))
        
        for i in range(n_features):
            return_arr[:, 2*i]   = np.cos(arr[:, i])
            return_arr[:, 2*i+1] = np.sin(arr[:, i])

        return return_arr


class AutoKMeans:
    """Performs automatic clustering using KMeans++ including dimensionality 
    reduction of the feature space.

    Arguments:
        data_directory (PathLike): Directory where data files can be found.
        pattern (str): Optional filename pattern to select out subset of npy files
            using glob. 
        dataloader (Type[_T]): Defaults to GenericDataLoader. Which dataloader to use.
        max_clusters (int): Defaults to 10. The maximum number of clusters to test
            during parameter sweep.
        stride (int): Defaults to 1. Linear stride of number of clusters during 
            parameter sweep. Aids on not testing too many values if number of clusters
            is high.
        reduction_algorithm (str): Defaults to PCA. Which dimensionality reduction
            algorithm to use. Currently only PCA is supported.
        reduction_kws (dict[str, Any]): Defaults to {'n_components': 2} for PCA. kwargs
            for supplied reduction_algorithm.
    """
    def __init__(self,
                 data_directory: PathLike,
                 pattern: str='',
                 dataloader: Type[_T]=GenericDataloader,
                 max_clusters: int=10,
                 stride: int=1,
                 reduction_algorithm: str='PCA',
                 reduction_kws: dict[str, Any]={'n_components': 2}):
        self.data_dir = Path(data_directory) 
        self.dataloader = dataloader(list(self.data_dir.glob(f'{pattern}*.npy')))
        self.data = self.dataloader.data
        self.shape = self.dataloader.shape
        
        self.n_clusters = max_clusters
        self.stride = stride

        self.decomposition = Decomposition(reduction_algorithm, **reduction_kws)
    
    def run(self) -> None:
        """Runs the automated clustering workflow.

        Returns:
            None
        """
        self.reduce_dimensionality()
        self.sweep_n_clusters([n for n in range(2, self.n_clusters, self.stride)])
        self.map_centers_to_frames()
        self.save_centers()
        self.save_labels()

    def reduce_dimensionality(self) -> None:
        """Performs dimensionality reduction using decomposer of choice.

        Returns:
            None
        """
        self.reduced = self.decomposition.fit_transform(self.data)

    def sweep_n_clusters(self,
                         n_clusters: list[int]) -> None:
        """Uses silhouette score to perform a parameter sweep over number of clusters.
        Stores the cluster centers for the best performing parameterization.

        Arguments:
            n_clusters (list[int]): List of number of clusters to test.

        Returns:
            None
        """
        best_centers = None
        best_labels = None
        best_score = 0.
        for n in tqdm(n_clusters, total=len(n_clusters), position=0, 
                      leave=False, desc='Sweeping `n_clusters`'):
            clusterer = KMeans(n_clusters=n)
            labels = clusterer.fit_predict(self.reduced)
            average_score = silhouette_score(self.reduced, labels)

            if average_score > best_score:
                best_centers = clusterer.cluster_centers_
                best_labels = labels
                best_score = average_score

        self.centers = best_centers
        self.labels = best_labels

    def map_centers_to_frames(self) -> None:
        """Finds and stores the data point which lies closest to the cluster center 
        for each cluster.

        Returns:
            None
        """
        cluster_centers = {i: None for i in range(len(self.centers))}
        for i, center in enumerate(self.centers):
            closest = 100.
            for p, point in enumerate(self.reduced):
                if (dist := np.linalg.norm(point - center)) < closest:
                    rep = p // self.shape[0]
                    frame = p % self.shape[0]
                    cluster_centers[i] = (rep, frame)
                    closest = dist

        self.cluster_centers = cluster_centers

    def save_centers(self) -> None:
        """Saves out cluster centers as a json file.

        Returns:
            None
        """
        with open(str(self.data_dir / 'cluster_centers.json'), 'w') as fout:
            json.dump(self.cluster_centers, fout, indent=4)

    def save_labels(self) -> None:
        """Generates a polars dataframe containing system, frame and cluster label 
        assignments and saves to a parquet file.

        Returns:
            None
        """
        files = self.dataloader.files
        if isinstance(self.dataloader.shape, tuple):
            shapes = [self.dataloader.shape[0]] * len(files)
        else:
            shapes = [shape[0] for shape in self.dataloader.shape]
        
        df = pl.DataFrame()
        for file, shape in zip(files, shapes):
            temp = pl.DataFrame({'system': [file.name] * shape, 'frame': np.arange(shape)})
            df = pl.concat([df, temp], how='vertical')

        df = df.with_columns(pl.Series(self.labels).alias('cluster'))

        df.write_parquet(str(self.data_dir / 'cluster_assignments.parquet'))


class Decomposition:
    """Thin wrapper for various dimensionality reduction algorithms. Uses scikit-learn style
    methods like `fit` and `fit_transform`.

    Arguments:
        algorithm (str): Which algorithm to use from PCA, TICA and UMAP. Currently only
            PCA is supported.
        kwargs: algorithm-specific kwargs to inject into the decomposer.
    """
    def __init__(self,
                 algorithm: str,
                 **kwargs):
        algorithms = {
            'PCA': PCA,
            'TICA': None,
            'UMAP': None
        }

        self.decomposer = algorithms[algorithm](**kwargs)
    
    def fit(self,
            X: np.ndarray) -> None:
        """Fits the decomposer with data.

        Arguments:
            X (np.ndarray): Array of input data.

        Returns:
            None
        """
        self.decomposer.fit(X)

    def transform(self,
                  X: np.ndarray) -> np.ndarray:
        """Returns the reduced dimension data from a decomposer which has 
        already been fit.

        Arguments:
            X (np.ndarray): Array of input data.

        Returns:
            (np.ndarray): Reduced dimension data.
        """
        return self.decomposer.transform(X)

    def fit_transform(self,
                      X: np.ndarray) -> np.ndarray:
        """Fits the decomposer with data and returns the reduced dimension
        data.

        Arguments:
            X (np.ndarray): Array of input data.

        Returns:
            (np.ndarray): Reduced dimension data.
        """
        return self.decomposer.fit_transform(X)
