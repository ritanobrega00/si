from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse


class KNNRegressor(Model):
    """
    KNNRegressor is suitable for regression problems. 
    Therefore, it estimates the average value of the k most similar examples instead of the most common class.
    
    Methods:
    _fit: stores the trainning dataset
    _predict: estimates the label value (y) for a sample based on the k most similar examples in the training dataset
    _score: calculates the error between the estimated values and the real ones (rmse) - accuracy metric

    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Parameters:
        k: int, default = 1 -> The number of nearest neighbors to use
        distance: Callable, default = euclidean_distance -> function that calculates the distance between a sample and samples in the training set

        Estimated Parameters:
        dataset: dataset object -> The training data
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Aim: Fit the model to the given dataset

        Parameters:
        dataset: Dataset object -> The dataset to fit the model

        Returns:
        self: KNNRegressor
        """
        self.dataset = dataset
        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Aim: Predict the label value (y) for a sample based on the k most similar examples in the training dataset

        Parameters:
        dataset: Dataset object -> The dataset to predict the label value (training)

        Returns:
        np.ndarray -> predictions –> an array of predicted values for the testing dataset (y_pred)
        """

        predictions = []
        for sample in dataset.X:
            # distance between the sample and various samples in the training dataset through the distance function
            dist = self.distance(sample, self.dataset.X)
            # Obtain the indexes of the k most similar examples (the ones with the smallest distance)
            k_indx = np.argsort(dist)[:self.k]
            # Get the corresponding values in Y:
            k_y = self.dataset.y[k_indx]
            # Calculate the average of the k most similar samples
            prediction = np.mean(k_y)
            #add the value to an empty list that will be then transformed into an array
            predictions.append(prediction)
        return np.array(predictions)
    
    def _score(self, dataset: Dataset) -> float:
        """
        Aim: Calculate the error between the estimated values and the real ones (rmse)

        Parameters:
        dataset: Dataset object 

        Returns:
        float -> error between predictions and actual values (rmse)  
        """
        #get the prediction values (y_pred)
        predictions = self._predict(dataset)
        #calculate the RMSE between the true and predicted values
        return rmse(dataset.y, predictions)
    
if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # Generate random dataset
    dataset_ = Dataset.from_random(600, 100, 1)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    knn = KNNRegressor(k=3)
    knn.fit(dataset_train)
    predictions = knn.predict(dataset_test)
    score = knn.score(dataset_test, predictions)
    print(f'The RMSE of the model is: {score}')

