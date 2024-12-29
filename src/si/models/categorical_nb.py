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
        super().__init__()
        self.smothing = smothing
        self.class_prior = None
        self.feature_probs = None
        self.classes = None

    def _fit(self, dataset: Dataset) -> 'CategoricalNB':
        n_samples, n_features = dataset.X.shape
        self.classes = np.unique(dataset.y)
        n_classes = len(self.classes)

        # Initialize arrays for class counts, feature counts, and class priors
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))

        # Compute class counts and feature counts per class
        for i in range(n_samples):
            class_idx = np.where(self.classes == dataset.y[i])[0][0]
            class_counts[class_idx] += 1
            feature_counts[class_idx] += dataset.X[i]

        # Compute class priors
        class_prior = class_counts / n_samples

        # Apply Laplace smoothing to feature counts
        feature_counts += self.smothing

        # Compute the feature probabilities (P(feature | class))
        feature_probs = feature_counts / (class_counts[:, np.newaxis] + 2 * self.smothing)

        # Store the computed values
        self.class_counts = class_counts
        self.feature_probs = feature_probs
        self.class_prior = class_prior

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        n_samples = dataset.X.shape[0]
        n_classes = len(self.classes)

        # Initialize an array for storing class probabilities
        class_probs = np.zeros((n_samples, n_classes))

        # Compute probabilities for each sample and each class
        for i in range(n_samples):
            sample = dataset.X[i]
            for c in range(n_classes):
                # Calculate the class prior
                prior = self.class_prior[c]

                # Calculate feature likelihood
                likelihood = np.prod(
                    sample * self.feature_probs[c] + (1 - sample) * (1 - self.feature_probs[c])
                )

                # Compute total probability for the class
                class_probs[i, c] = prior * likelihood

        # Return the class with the highest probability
        return self.classes[np.argmax(class_probs, axis=1)]

    def _score(self, dataset: Dataset, predictions: np.array) -> float:
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)