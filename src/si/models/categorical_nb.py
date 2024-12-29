import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse

class CategoricalNB(Model):
    """
    CategoricalNB is a Naive Bayes classifier suitable for classification problems with categorical features.
        Parameters:
        smothing: float, default = 1.0 -> Laplace smoothing to avoid zero probabilities

        Estimated Parameters:
        class_prior: list -> The prior probability of each class
        feature_probs: list -> Probabilities for each feature for each class being present/being 1

        Methods:
        fit: estimates the class_prior and feature_probs from the Dataset
        predict: Predict the class labels for a given set of samples
       score: calculates the error between the estimated classes and the actual ones
    """

    def __init__(self, smothing=1.0):
        self.smothing = smothing

    def _fit(self, dataset: Dataset) -> 'CategoricalNB':
        n_samples, n_features = dataset.X.shape
        n_classes = len(np.unique(dataset.y))

        # Initialize arrays for class counts, feature counts, and class priors
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        class_prior = np.zeros(n_classes)

        self.classes, self.class_counts = np.unique(y, return_counts=True)
        self.class_prior = self.class_counts / len(y)
        self.feature_counts = np.array([np.bincount(X[y == c].flatten(), minlength=X.shape[1]) for c in self.classes])
        self.feature_prob = (self.feature_counts + self.alpha) / (self.class_counts[:, None] + self.alpha * X.shape[1])
        return self

    def _predict(self, X):
        return self.classes[np.argmax(np.log(self.class_prior) + X @ np.log(self.feature_prob).T, axis=1)]