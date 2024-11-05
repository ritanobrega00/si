def train_test_split(dataset, test_size=0.2, random_state=None)->tuple:
    """
    Split the dataset into a training and a test set.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float, optional
        The proportion of the dataset to include in the test split.
    random_state: int, optional
        The seed used by the random number generator.

    Returns
    -------
    train_set: Dataset
        The training set.
    test_set: Dataset
        The test set.
    """
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(dataset))
    test_size = int(len(dataset) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return dataset[train_indices], dataset[test_indices]