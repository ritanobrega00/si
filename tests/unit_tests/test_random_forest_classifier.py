import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np
from datasets import DATASETS_PATH

from si.io.data_file import read_data_file
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier 
from si.models.random_forest_classifier import RandomForestClassifier


class TestRandomForestClassifier(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12],
                           [13, 14, 15]])
        self.y = np.array([0, 1, 0, 1, 0])
        self.dataset_custom = Dataset(self.X, self.y)

        self.csv_iris = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset_iris = read_csv(filename=self.csv_iris, features=True, label=True)
        self.csv_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_cpu, features=True, label=True)
        self.csv_bb = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset_bb = read_data_file(filename=self.csv_bb, label=True, sep=",")

        self.datasets = [self.dataset_iris, self.dataset_cpu, self.dataset_bb, self.dataset_custom]
        
        self.

    def test_fit(self):
        self.random_forest.fit(self.dataset)
        self.assertEqual(len(self.random_forest.trees), 5)
        for tree, features in self.random_forest.trees:
            self.assertIsInstance(tree, DecisionTreeClassifier)
            self.assertEqual(len(features), 2)

    def test_predict(self):
        self.random_forest.fit(self.dataset)
        predictions = self.random_forest.predict(self.dataset)
        self.assertEqual(len(predictions), len(self.dataset.y))
        self.assertTrue(set(predictions).issubset(set(self.dataset.y)))

    def test_score(self):
        self.random_forest.fit(self.dataset)
        predictions = self.random_forest.predict(self.dataset)
        score = self.random_forest.score(self.dataset, predictions)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_different_max_features(self):
        rf = RandomForestClassifier(n_estimators=5, max_features=3, min_sample_split=2, max_depth=3, mode='gini', seed=42)
        rf.fit(self.dataset)
        self.assertEqual(len(rf.trees), 5)
        for tree, features in rf.trees:
            self.assertEqual(len(features), 3)

    def test_different_n_estimators(self):
        rf = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=3, mode='gini', seed=42)
        rf.fit(self.dataset)
        self.assertEqual(len(rf.trees), 10)

    def test_different_modes(self):
        rf_gini = RandomForestClassifier(n_estimators=5, max_features=2, min_sample_split=2, max_depth=3, mode='gini', seed=42)
        rf_entropy = RandomForestClassifier(n_estimators=5, max_features=2, min_sample_split=2, max_depth=3, mode='entropy', seed=42)
        rf_gini.fit(self.dataset)
        rf_entropy.fit(self.dataset)
        self.assertEqual(len(rf_gini.trees), 5)
        self.assertEqual(len(rf_entropy.trees), 5)

    def test_random_seed(self):
        # Test if the seed parameter ensures reproducibility
        rf1 = RandomForestClassifier(n_estimators=3, seed=42)
        rf2 = RandomForestClassifier(n_estimators=3, seed=42)

        # Ensure that two models with the same seed produce the same results
        self.assertEqual(rf1._fit(self.dataset).trees, rf2._fit(self.dataset).trees)

if __name__ == '__main__':
    unittest.main()

