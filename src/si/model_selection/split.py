from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> tuple[Dataset, Dataset]:
    """
    Aim: Split the dataset into stratified training and testing sets

    Parameters:
    dataset: object Dataset -> The dataset to split into train and test datasets
    test_size: float, default = 0.2 -> The size of the dataset to include in the test split (e.g. 0.2 for 20%)
    random_state: int, default = 42 -> The seed of the random number generator

    Returns a tuple of two datasets:
    train: Dataset -> The stratified trainning dataset
    test: Dataset ->  The stratified testing dataset
    """
    np.random.seed(random_state)

    # Get unique class labels and their counts
    unique_classes, class_counts = np.unique(dataset.y, return_counts=True)

    train_idxs = []
    test_idxs = []

    # Perform the stratified split for each class label
    for classes, counts in zip(unique_classes, class_counts):
        # Get indices of samples with the current class
        idxs = np.where(dataset.y == classes)[0]
        # Calculate the number of test samples for the current class
        n_test = int(np.floor(counts * test_size))
        # Shuffle the indices
        np.random.shuffle(idxs)
        # Select indices for the current class and add them to the test indices and the remaining to the train indices
        train_idxs.extend(idxs[n_test:])
        test_idxs.extend(idxs[:n_test])
    
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test