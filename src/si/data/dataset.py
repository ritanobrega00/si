from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)
    
    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    def dropna(self) -> 'Dataset':
        """
        Aim: Removes all samples from the Dataset where at least one independent feature contains a NaN
        Updates the y vector by removing the entry associated with the sample to be removed
        Arguments: None
        Returns: modified Dataset
        """
        # Identification of the rows with no NaN values, using a mask
        mask = ~np.isnan(self.X).any(axis=1)  
        
        # The mask is used to filter rows in self.X where no NaN values are present
        self.X = self.X[mask]
        
        # If there is a y  vector (is not None) 
        # then its updated by removing entries associated with the samples to be eliminated
        if self.y is not None:
            self.y = self.y[mask]
        return self
    
    def fillna(self, value: Union[int, float]) -> 'Dataset':
        """
        Aim: Replaces all NaN values in the Dataset with the specified value (mean or median of the feature/variable)
        Ensures that no NaN values remain in the modified Dataset and raises an error otherwise.
        Arguments: value (int or float) - the value to fill NaN with
        Returns: modified Dataset
        """
        # first we check if the value is a float or an int, if is is then we replace NaN with the specified value
        if isinstance(value, float):
            # Replace NaN with the specified value
            self.X = np.where(np.isnan(self.X), value, self.X)

        # if the value is a string, we check if it is 'mean' or 'median', and then replace NaN according to the input given
        elif value == "mean":
            # Replace NaN with the mean of each column
            means = np.nanmean(self.X, axis=0)
            inds = np.where(np.isnan(self.X))
            self.X[inds] = np.take(means, inds[1])
        elif value == "median":
            # Replace NaN with the median of each column
            medians = np.nanmedian(self.X, axis=0)
            inds = np.where(np.isnan(self.X))
            self.X[inds] = np.take(medians, inds[1])
        else:
            raise ValueError("Invalid value parameter. Use a float, 'mean', or 'median'.")            
        return self
    
    def remove_by_index(self, index: int) -> 'Dataset':
        """
        Aim: Removes the sample at the specified index from the Dataset
        Updates the y by removing the entry associated with the sample to be removed
        Arguments: index (int) - the index of the sample to remove
        Returns: modified Dataset
        """
        # Remove the sample at the specified index
        self.X = np.delete(self.X, index, axis=0)
        
        # If there is a y vector (is not None) then its updated by removing the entry associated with the sample to be removed
        if self.y is not None:
            self.y = np.delete(self.y, index)
        return self

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
