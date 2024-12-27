import numpy as np
from si.decomposition.transformer import Transformer

class PCA(Transformer):
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, X):
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Calculate the covariance matrix and perform eigenvalue decomposition
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 3: Infer the principal components
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]

        # Step 4: Infer the explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance

    def _transform(self, X):
        # Step 1: Center the data
        X_centered = X - self.mean

        # Step 2: Calculate the reduced X
        X_reduced = np.dot(X_centered, self.components)
        return X_reduced