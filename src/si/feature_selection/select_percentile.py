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
        #if the percentile is 0, no features are selected, so it raises an error
        if self.percentile == 0:
            raise ValueError("Cannot perform percentile selection with a percentile of 0%.")
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Aim: Estimates the F and p values for each feature using the scoring function.
        Parameters: dataset (object Dataset)
        Returns self (the fitted SelectPercentile object)
        """        
        #if the dataset only has one feature, an error is raised
        if len(dataset.features) == 1:
            raise ValueError("Cannot perform percentile selection with a single feature.")
        else:
            self.F, self.p = self.score_func(dataset)
            return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Aim: selects features with the highest F value up to the specified percentile
        Parameters: dataset (Dataset object)
        Returns: a Dataset object with the selected features (object Dataset)
        """
        #Check if fit has been called before
        if self.F is None:
            raise ValueError("The transformer must be fitted before calling transform.")
        
        num_features = int(len(self.F) * (self.percentile / 100))  # number of features to select based on the percentile
        # if the percentile is too low to select any features, an error is raised
        if num_features < 1:
            raise ValueError("Percentile too low to select any features.")
        # Since you cannot select a decimal number of features, the result is usually rounded down
        else:
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