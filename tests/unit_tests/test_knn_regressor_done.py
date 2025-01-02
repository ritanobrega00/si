import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split
from si.models.knn_regressor import KNNRegressor


class TestKNNRegressor(unittest.TestCase):

    def setUp(self):
        self.csv_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_cpu, features=True, label=True)

        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        self.dataset_random = Dataset(X, y)

        self.datasets = [self.dataset_cpu, self.dataset_random]

        self.k_values = [1, 3, 5]

    def test_k_value(self):
        with self.assertRaises(ValueError):
            KNNRegressor(k=0)
        with self.assertRaises(ValueError):
            KNNRegressor(k=-1)

    def test_fit(self):
        for dataset in self.datasets:
            train_normal, test_normal = train_test_split(dataset, test_size=0.2, random_state=42)
            for k in self.k_values:
                knn = KNNRegressor(k=k)
                fitted_dataset = knn.fit(train_normal)
                self.assertTrue(isinstance(train_normal, Dataset))
                self.assertTrue(isinstance(fitted_dataset, KNNRegressor))
                self.assertTrue(np.all(train_normal.X == fitted_dataset.dataset.X))
                self.assertTrue(np.all(train_normal.y == fitted_dataset.dataset.y))

    def test_predict(self):
        for dataset in self.datasets:
            train_normal, test_normal = train_test_split(dataset, test_size=0.2, random_state=42)
            for k in self.k_values:
                knn = KNNRegressor(k=k)
                knn.fit(train_normal)
                predictions = knn.predict(test_normal)
                
                self.assertTrue(isinstance(predictions, np.ndarray))
                self.assertEqual(predictions.shape[0], test_normal.y.shape[0])

    def test_score(self):
        for dataset in self.datasets:
            train_normal, test_normal = train_test_split(dataset, test_size=0.2, random_state=42)
            for k in self.k_values:
                knn = KNNRegressor(k=k)
                knn.fit(train_normal)
                knn.predict(test_normal)
                model_score = knn.score(test_normal)
                self.assertTrue(isinstance(model_score, float))
                self.assertTrue(model_score >= 0)



if __name__ == '__main__':
    unittest.main()
