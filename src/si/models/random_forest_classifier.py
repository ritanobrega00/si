import numpy as np
from typing import Tuple, List, Union, Literal
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier  

class RandomForestClassifier(Model):
    """
    Random Forest Classifier.

    Parameters
    - n_estimators: int, default=10 – number of decision trees to use
    - max_features: int, default=None (if not specified, does not exist) – maximum number of features to use per tree
    - min_sample_split: int, default=2 - minimum samples allowed in a split
    - max_depth: int, default=10 – maximum depth of the trees
    - mode: between the gini or entropy, the default is the gini  – impurity calculation mode 
    - seed: int, default=None – random seed to use to assure reproducibility

    Estimated Parameters
    - trees – list of tuples that are composed of the the trees of the random forest and respective features 

    Methods
    - fit: train the decision trees of the random forest
    - predict: predict the labels using the ensemble models
    - score: compute the accuracy of the random forest
    """

    def __init__(self, n_estimators: int = 10, max_features: int = None, min_sample_split: int = 2,
                 max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini', seed: int = None, **kwargs) -> None:
        """
        Initializes the RandomForestClassifier.

        Parameters
        ----------
        n_estimators: int
            The number of decision trees to use.
        max_features: int
            The maximum number of features to use per tree.
        min_sample_split: int
            The minimum number of samples required to split an internal node.
        max_depth: int
            The maximum depth of the trees.
        mode: Literal['gini', 'entropy']
            The impurity calculation mode ('gini' or 'entropy').
        seed: int
            The random seed for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # List to store (trained decision tree, respective features) tuples (initially empty)
        self.trees: List[Tuple[DecisionTreeClassifier, np.ndarray]] = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train the Random Forest by creating multiple Decision Trees on bootstrap samples.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        # Set random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.shape()

        # Define the maximum number of features if not explicitly provided
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))  # Standard practice for classification problems

        # Create an iteration to apply to all the trees in the forest
        for _ in range(self.n_estimators):
            # Creation of a Bootstrap dataset - random samples with replacement
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_X = dataset.X[sample_indices]
            bootstrap_y = dataset.y[sample_indices]

            # Select random features without replacement
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            bootstrap_X = bootstrap_X[:, feature_indices]

            # Create and Train a decision tree with the bootstrap dataset
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth, mode=self.mode)
            tree.fit(Dataset(bootstrap_X, bootstrap_y))

            # Step 5: Store the tree and the feature indices
            self.trees.append((tree, feature_indices))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Aim: Predict the labels using the ensemble of decision trees.

        Parameters:
        dataset: Dataset - The dataset to make predictions for.

        Returns: np.ndarray - The predicted labels
        """
        predictions = []

        for tree, feature_indices in self.trees:
            # Use only the features that the tree was trained on
            subset_X = dataset.X[:, feature_indices]
            predictions.append(tree.predict(Dataset(subset_X, dataset.y)))

        # Transpose predictions to shape (n_samples, n_trees)
        predictions = np.array(predictions).T

        # get the most common prediction for each sample
        final_predictions = [np.bincount(row).argmax() for row in predictions]

        # If the labels are strings (like in the iris.csv), map back to original labels
        if isinstance(dataset.y[0], str):
            unique_labels = np.unique(dataset.y)
            final_predictions = unique_labels[final_predictions]

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Aim: Compute the accuracy of the Random Forest.

        Parameters:
        dataset: Dataset - The dataset to compute the score for
        predictions: np.ndarray - The predicted labels

        Returns: float - The accuracy score
        """ 
        score = accuracy(dataset.y, predictions)
        return score
