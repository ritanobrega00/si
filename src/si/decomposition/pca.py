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
    components: array, shape (n_features, n_components) - the principal components (a matrix where each row is an
eigenvector corresponding to a principal component)
    explained_variance: array, shape (n_components,) - the amount of variance explained by each principal
component (it's a vector of eigenvalues)

    Methods:
    _fit – estimates the mean, principal components, and explained variance
    _transform – calculates the reduced dataset using the principal components
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, X):
        # Centering the data (get the mean and subtract it from the data for each feature)
        self.mean = np.mean(X, axis=0) #axis=0 means that the mean is calculated for each feature/column across all samples
        X_centered = X - self.mean

        # Calculation of the covariance matrix of the centered data
        covariance_matrix = np.cov(X_centered, rowvar=False) # rowvar=False means that the columns are the variables and rows are samples
        # Eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Infer the principal components that correspond to the n (nº components) highest eigenvalues
        idx = np.argsort(eigenvalues)[::-1] #sorting in descending order to get the indices of the highest eigenvalues
        eigenvalues = eigenvalues[idx]      #reordering the eigenvalues
        eigenvectors = eigenvectors[:, idx] #reordering the columns of the eigenvectors matrix to match the eigenvalues
        self.components = eigenvectors[:, :self.n_components] #selecting the first n_components eigenvectors that become the principal components

        # Infer the explained variance (dividing the eigenvalue by the sum of all eigenvalues)
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance

    def _transform(self, X):
        # Centering the data (using what was done previously in the fit method)
        X_centered = X - self.mean

        # Reduction of X to the principal components = X * matrix of principal components
        X_reduced = np.dot(X_centered, self.components)
        return X_reduced