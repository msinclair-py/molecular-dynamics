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
    """
    Loads any generic data stored in numpy arrays and stores the full
    dataset. Capable of loading data with variable row lengths but must
    be consistent in the columnar dimension.
    """
    def __init__(self,
                 data_files: list[PathLike]):
        self.files = data_files
        self.load_data()

    def load_data(self):
        """
        Loads data into one large array.
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
    def data(self):
        return self.data_array

    @property
    def shape(self):
        if len(set(self.shapes)) == 1:
            return self.shapes[0]
        
        return self.shapes


class PeriodicDataloader(GenericDataloader):
    """
    Decomposes periodic data using sin and cos, returning double the features.
    """
    def __init__(self,
                 data_files: list[PathLike]):
        super().__init__(data_files)
        
    def load_data(self):
        """
        Loads file of input periodic data.
        """
        self.data_array = []
        self.shapes = []
        for f in self.files:
            temp = self.remove_periodicity(np.load(str(f)))
            
            self.shapes.append(temp.shape)
            self.data_array.append(temp)

    def remove_periodicity(self,
                           arr: np.ndarray) -> np.ndarray:
        """
        Removes periodicity from each feature using sin and cos.
        """
        n_features = arr.shape[1]
        return_arr = np.zeros((arr.shape[0], n_features * 2))
        
        for i in range(n_features):
            return_arr[:, 2*i]   = np.cos(arr[:, i])
            return_arr[:, 2*i+1] = np.sin(arr[:, i])

        return return_arr


class AutoKMeans:
    """
    Performs automatic clustering using KMeans++ including dimensionality reduction
    of the feature space.
    """
    def __init__(self,
                 data_directory: PathLike,
                 pattern: str='',
                 dataloader: Type[_T]=GenericDataloader,
                 max_clusters: int=10,
                 stride: int=1,
                 reduction_algorithm: str='PCA',
                 reduction_kws: dict[str, Any]={'n_components': 2}):
        self.data_dir = Path(data_directory) if isinstance(data_directory, str) else data_directory
        self.dataloader = dataloader(list(self.data_dir.glob(f'{pattern}*.npy')))
        self.data = self.dataloader.data
        self.shape = self.dataloader.shape
        
        self.n_clusters = max_clusters
        self.stride = stride

        self.decomposition = Decomposition(reduction_algorithm, **reduction_kws)
    
    def run(self) -> None:
        """
        Runs the automated clustering workflow.
        """
        self.reduce_dimensionality()
        self.sweep_n_clusters([n for n in range(2, self.n_clusters, self.stride)])
        self.map_centers_to_frames()
        self.save_centers()
        self.save_labels()

    def reduce_dimensionality(self) -> None:
        """
        Performs dimensionality reduction using decomposer of choice.
        """
        self.reduced = self.decomposition.fit_transform(self.data)

    def sweep_n_clusters(self,
                         n_clusters: list[int]) -> None:
        """
        Uses silhouette score to perform a parameter sweep over number of clusters.
        Stores the cluster centers for the best performing parameterization.
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
        """
        Finds and stores the data point which lies closest to the cluster center 
        for each cluster.
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
        """
        Saves out cluster centers as a json file.
        """
        with open(str(self.data_dir / 'cluster_centers.json'), 'w') as fout:
            json.dump(self.cluster_centers, fout, indent=4)

    def save_labels(self) -> None:
        """
        Generates a polars dataframe containing system, frame and cluster label 
        assignments and saves to a parquet file.
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
    """
    Thin wrapper for various dimensionality reduction algorithms. Uses scikit-learn style
    methods like `fit` and `fit_transform`.
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
        self.decomposer.fit(X)

    def transform(self,
                  X: np.ndarray) -> np.ndarray:
        return self.decomposer.transform(X)

    def fit_transform(self,
                      X: np.ndarray) -> np.ndarray:
        return self.decomposer.fit_transform(X)
