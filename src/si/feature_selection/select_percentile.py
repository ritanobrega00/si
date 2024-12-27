import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.base.transformer import Transformer

class SelectPercentile(Transformer):
    """
    Aim: Select features according to the percentile.
    
    Parameters:
    score_func: variance analysis function (f_classification by default)
    percentile: int, default=10

    Estimated parameters (value for each feature estimated by score_func): 
    F: array, shape (n_features,)
    p: array, shape (n_features,)
    """

    def __init__(self, score_func=f_classification, percentile=10.0):
        """
        Aim: Select features according to the percentile.
        
        Parameters:
        score_func: variance analysis function (f_classification by default)
        percentile: float, percentile of features to select (default=10.0)
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset):
        """
        Aim: Estimates the F and p values for each feature using the scoring function.
        Parameters: dataset (object Dataset)
        Returns self (the fitted SelectPercentile object)
        """        
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset):
        """
        Aim: selects features with the highest F value up to the specified percentile
        Parameters: dataset (Dataset object)
        Returns: a Dataset object with the selected features (object Dataset)
        """
        num_features = int(len(self.F) * (self.percentile / 100)) # number of features to select based on the percentile
        top_indices = np.argsort(self.F)[-num_features:]  #indices of the features with higher F value
        new_dataset = Dataset(X=dataset.X[:, top_indices], y=dataset.y, features=[dataset.features[i] for i in top_indices])
        return new_dataset
    
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = SelectPercentile(percentile=50)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)