import numpy as np
from si.base.transformer import Transformer
class PCA(Transformer):
    """
    Aim: perform a PCA, reducing the dimension of the dataset, 
using eigenvalue decomposition of the covariance matrix 
    
    Parameters:
    n_components: int, number of components to keep
    
    Estimated parameters:
    mean: array, shape (n_features,) - mean of the samples
    components: array, shape (n_components, n_features) - the principal components (a matrix where each row is an
eigenvector corresponding to a principal component)
    explained_variance: array, shape (n_components,) - the amount of variance explained by each principal
component (it's a vector of eigenvalues)

    Methods:
    _fit – estimates the mean, principal components, and explained variance
    _transform – calculates the reduced dataset using the principal components
    """
    def __init__(self, n_components:int):
        if n_components <= 0:
            raise ValueError("n_components must be greater than 0.")
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.X_normalized =False

    def _fit(self, X: np.array, normalization:bool=False) -> None:
        """
        the _fit method doesn't return anything, it just stores the estimated parameters in the object
        first we check/transform the input data to a numpy array the we check if the number of components is valid

        Centering the data is an essential step in PCA, however we centering and normalizing are not the same thing
        Therefore if the user wants to normalize the data, one of the parameters will be normalization = True (otherwise it will just be centered)
        If this parameter equals True, there will be an extra step to scale the data to have a unit variance for each feature 
        """
        if self.n_components > X.shape[1]:
            raise ValueError(f"n_components must be in the range (0, {X.shape[1]}].")

        # Centering the data (get the mean and subtract it from the data for each feature)
        self.mean = np.mean(X, axis=0) #axis=0 means that we are calculating the mean for each feature
        X_centered = X - self.mean

        # Normalizing the data (scaling the data by dividing it by the standard deviation of each feature)
        if normalization:
            X_centered /= np.std(X, axis=0)
            self.X_normalized = True
        else:
            self.X_normalized = False
        
        # Calculation of the covariance matrix of the centered data
        covariance_matrix = np.cov(X_centered, rowvar=False) # rowvar=False means that the columns are the variables and rows are samples
        # Eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Infer the principal components that correspond to the n (nº components) highest eigenvalues
        idx = np.argsort(eigenvalues)[::-1] #sorting in descending order to get the indices of the highest eigenvalues
        eigenvalues = eigenvalues[idx]      #reordering the eigenvalues
        eigenvectors = eigenvectors[:, idx] #reordering the columns of the eigenvectors matrix to match the eigenvalues
        # Store the first n_components eigenvectors that become the principal components
        self.components = eigenvectors[:, :self.n_components].T

        # Infer the explained variance (dividing the eigenvalue by the sum of all eigenvalues)
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance


    def _transform(self, X: np.array) -> np.array:
        # as the transform method is going to use the stored estimated parameters from the fit method
        # We perform a check to ensure that the fit method has been called before calling the transform
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA has not been fitted. Call fit() before transform().")
        
        # Centering the data
        X_centered = X - self.mean

        #if the normalization has been selected in the fit method
        if self.X_normalized:
            X_centered /= np.std(X, axis=0)
        
        # Reduction of X to the principal components = X * matrix of principal components   
        X_reduced = np.dot(X_centered, self.components.T)
        return X_reduced
    
if __name__ == "__main__":
    import sys
    import os 
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from datasets import DATASETS_PATH

    from si.decomposition.pca import PCA
    from si.io.csv_file import read_csv

    csv_iris = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
    iris_dataset = read_csv(filename=csv_iris, features=True, label=True)
    pca_norm = PCA(n_components=2)
    pca_norm._fit(iris_dataset.X, normalization=True)
    X_transformed_norm = pca_norm._transform(iris_dataset.X)
    print('Explained variance (normalized data):', pca_norm.explained_variance)
    print('Transformed data structure (normalized data):', X_transformed_norm.shape)
    print('Components (normalized data):', pca_norm.components)

    pca = PCA(n_components=2)
    pca._fit(iris_dataset.X)
    X_transformed = pca._transform(iris_dataset.X)
    print('Explained variance with data not normalized:', pca.explained_variance)
    print('Transformed data structure with data not normalized:', X_transformed.shape)
    print('Components (not normalized data):', pca.components)