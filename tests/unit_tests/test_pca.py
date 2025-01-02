import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from sklearn.decomposition import PCA as PCA_sklearn

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.csv_bb = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset_bb = read_csv(filename=self.csv_bb, features=True, label=False)
        self.csv_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_cpu, features=True, label=True)
        self.datasets = [self.dataset_bb, self.dataset_cpu]

        self.different_n_comp = [2, 3, 4, 5]
    
    def test_pca_fit(self):
        for component in self.different_n_comp:
                pca = PCA(n_components=component)
                for dataset in self.datasets:
                    pca.fit(dataset)
                    self.assertIsNotNone(pca.mean)
                    self.assertIsNotNone(pca.components)
                    self.assertIsNotNone(pca.explained_variance)
                    self.assertEqual(pca.components.shape[1], component)
        #Test for n_components = 0
        pca_0_components = PCA(n_components=0)
        with self.assertRaises(ValueError):
            pca_0_components.fit(self.dataset_bb)
            
    def test_pca_transform(self):
        for component in self.different_n_comp:
                pca = PCA(n_components=component)
                for dataset in self.datasets:
                    pca.fit(dataset)
                    transformed_data = pca.transform(dataset)
                    self.assertEqual(transformed_data.shape[1], component)
                    self.assertEqual(transformed_data.shape[0], dataset.shape[0])
        # Test for a non-fitted PCA
        pca_non_fit = PCA(n_components=2)
        with self.assertRaises(AssertionError):
            pca_non_fit.transform(self.dataset_bb)

            
        def test_pca_fit_transform(self):
            for dataset in self.datasets:
                pca = PCA(n_components=2)
                pca._fit(dataset)
                transformed_data = pca._transform(dataset)
                self.assertEqual(transformed_data.shape[1], 2)
                self.assertEqual(transformed_data.shape[0], dataset.shape[0])
                self.assertIsNotNone(pca.mean)
                self.assertIsNotNone(pca.components)
                self.assertIsNotNone(pca.explained_variance)
        
        def test_normalization:
            pass


if __name__ == '__main__':
    unittest.main()