import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.preprocessing import StandardScaler 

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.csv_bb = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset_bb = read_csv(filename=self.csv_bb, features=True, label=False)
        self.csv_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_cpu, features=True, label=True)
        self.datasets = [self.dataset_bb, self.dataset_cpu]

        self.different_n_comp = [2, 3, 4, 5]
    
    def test_invalid_components(self):
        for c in [-1, 0]:
            with self.assertRaises(ValueError):
                PCA(n_components=c)

    def test_pca_fit(self):
        for component in self.different_n_comp:
            pca = PCA(n_components=component)
            for dataset in self.datasets:
                pca.fit(dataset.X)
                self.assertIsNotNone(pca.mean)
                self.assertIsNotNone(pca.components)
                self.assertIsNotNone(pca.explained_variance)
                self.assertEqual(pca.components.shape[0], component)
                self.assertEqual(pca.explained_variance.shape[0], component)

    def test_pca_normalization(self):
        for component in self.different_n_comp:
            pca = PCA(n_components=component)
            pca_norm = PCA(n_components=component)
            for dataset in self.datasets:
                data_scaled = StandardScaler().fit_transform(dataset.X)

                pca.fit(data_scaled)
                pca_norm._fit(dataset.X, normalization=True)
                
                self.assertIsNotNone(pca_norm.mean)
                self.assertIsNotNone(pca_norm.components)
                self.assertIsNotNone(pca_norm.explained_variance)
                self.assertEqual(pca_norm.components.shape[0], component)
                self.assertEqual(pca_norm.explained_variance.shape[0], component)

                #Test that the results are the same with normalization and with already normalized data
                self.assertTrue(np.allclose(pca_norm.explained_variance, pca.explained_variance))
                self.assertTrue(np.allclose(pca_norm.components, pca.components))           
            
    def test_pca_transform(self):
        for component in self.different_n_comp:
            pca = PCA(n_components=component)
            for dataset in self.datasets:
                pca.fit(dataset.X)
                transformed_data = pca.transform(dataset.X)

                self.assertEqual(transformed_data.shape, (dataset.X.shape[0], component))
                
        # Test for a non-fitted PCA
        pca_non_fit = PCA(n_components=2)
        with self.assertRaises(RuntimeError):
            pca_non_fit.transform(self.dataset_bb.X)


    def test_pca_sklearn(self):
        for component in self.different_n_comp:
            pca = PCA(n_components=component)
            pca_sklearn = PCA_sklearn(n_components=component)
            for dataset in self.datasets:
                pca.fit(dataset.X)
                pca_sklearn.fit(dataset.X)

                X_transformed = pca.transform(dataset.X)
                X_sklearn = pca_sklearn.transform(dataset.X)

                self.assertTrue(np.allclose(pca_sklearn.explained_variance_ratio_, pca.explained_variance))
                self.assertTrue(np.allclose(pca_sklearn.components_.shape, pca.components.shape))
                self.assertIsInstance(X_transformed, np.ndarray)
                self.assertEqual(X_transformed.shape, X_sklearn.shape)

                      


if __name__ == '__main__':
    unittest.main()