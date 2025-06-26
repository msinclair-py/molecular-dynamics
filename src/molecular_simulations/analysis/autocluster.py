from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Any, Union

PathLike = Union[Path, str]

class AutoKMeans:
    def __init__(self,
                 data_directory: PathLike,
                 pattern: str='',
                 dataloader: Callable=GenericDataLoader,
                 reduction_algorithm: str='PCA',
                 reduction_kws: dict[str, Any]):
        self.data_dir = Path(data_directory) if isinstance(data_directory, str) else data_directory
        self.dataloader = dataloader(self.data_dir.glob(f'{pattern}*.npy'))
        self.data = self.dataloader.data
        self.shape = self.dataloader.shape
        self.decomposition = Decomposition(reduction_algorithm, **reduction_kws)
    
    def run(self) -> None:
        pass

    def reduce_dimensionality(self) -> None:
        self.reduced = self.decomposition.fit_predict(self.data)

    def sweep_n_clusters(self,
                         n_clusters: list[int]) -> None:
        best_centers = None
        best_score = 0.
        for n in n_clusters:
            clusterer = KMeans(n_clusters=n)
            labels = clusterer.fit_predict(self.reduced)
            average_score = silhouette_score(self.reduced, labels)

            if average_score > best_score:
                best_centers = clusterer.cluster_centers_
                best_score = average_score

        self.centers = best_centers

    def map_centers_to_frames(self,
                              centers: np.ndarray) -> None:
        cluster_centers = {i: None for i in range(len(centers))}
        for i, center in enumerate(centers):
            closest = 100.
            for p, point in enumerate(self.reduced):
                if (dist := np.linalg.norm(point - center)) < closest:
                    rep = p // self.shape[0]
                    frame = p % self.shape[0]
                    cluster_centers[i] = (rep, frame)
                    closest = dist

        self.cluster_centers = cluster_centers

class GenericDataloader:
    def __init__(self,
                 data_files: list[PathLike]):
        self.files = data_files
        self.load_data()

    def load_data(self):
        data_array = []
        self.shapes = []
        for f in self.files:
            temp = np.load(str(f))
            self.shapes.append(temp.shape)
            data_array.append(temp)

        self.data_array = np.vstack(data_array)

    @property
    def data:
        return self.data_array

    @property
    def shape:
        return self.shapes

class DihedralDataloader:
    def __init__(self):
        pass

class Decomposition:
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
            X: np.ndarray) -> np.ndarray:
        return self.decomposer.fit(X)

    def fit_predict(self,
                    X: np.ndarray) -> np.ndarray:
        return self.decomposer.fit_predict(X)
