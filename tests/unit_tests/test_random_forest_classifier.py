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
        self.csv_iris = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset_iris = read_csv(filename=self.csv_iris, features=True, label=True)

        self.iris_train, self.iris_test = train_test_split(self.dataset_iris, test_size=0.2, random_state=42)

        self.random_forest = RandomForestClassifier(n_estimators=5, max_features=2,
                                                    min_sample_split=2, max_depth=10,
                                                    mode='gini', seed=42)

    def test_fit(self):
        fitted = self.random_forest.fit(self.iris_train)
        self.assertIsInstance(fitted, RandomForestClassifier)
        self.assertEqual(self.random_forest.n_estimators, 5)
        self.assertEqual(len(self.random_forest.trees), 5)
        for tree, features in self.random_forest.trees:
            self.assertIsInstance(tree, DecisionTreeClassifier)
            self.assertEqual(len(features), 2)

    def test_predict(self):
        self.random_forest.fit(self.iris_train)
        predictions = self.random_forest.predict(self.iris_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.iris_test.y))
                
    def test_score(self):
        self.random_forest.fit(self.iris_train)
        score = self.random_forest.score(self.iris_test)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_different_n_estimators(self):
        for n in [3, 5, 10]:
            rf = RandomForestClassifier(n_estimators=n, max_features=3, min_sample_split=2,
                                         max_depth=10, mode='gini', seed=42)
            rf.fit(self.iris_train)
            self.assertEqual(len(rf.trees), n)
            for tree, features in rf.trees:
                self.assertEqual(len(features), 3)
                

    def test_different_n_estimators(self):
        for n_feat in [2, 5, 10]:
            rf = RandomForestClassifier(n_estimators=8, max_features=n_feat, min_sample_split=2,
                                         max_depth=10, mode='gini', seed=42)
            rf.fit(self.iris_train)
            for tree, features in rf.trees:
                self.assertEqual(len(rf.trees), 8)
                self.assertEqual(len(features), n_feat)

    def test_different_modes(self):
        rf_gini = RandomForestClassifier(n_estimators=5, max_features=2, min_sample_split=2,
                                        max_depth=3, mode='gini', seed=42)
        rf_entropy = RandomForestClassifier(n_estimators=5, max_features=2, min_sample_split=2,
                                        max_depth=3, mode='entropy', seed=42)
        rf_gini.fit(self.iris_train)
        rf_entropy.fit(self.iris_train)
        self.assertEqual(len(rf_gini.trees), 5)
        self.assertEqual(len(rf_entropy.trees), 5)

    def test_random_seed(self):
        rf1 = RandomForestClassifier(n_estimators=3, seed=42)
        rf2 = RandomForestClassifier(n_estimators=3, seed=42)
        
        rf1.fit(self.dataset_iris)
        rf2.fit(self.dataset_iris)

        self.assertEqual(rf1.trees, rf2.trees)

if __name__ == '__main__':
    unittest.main()

