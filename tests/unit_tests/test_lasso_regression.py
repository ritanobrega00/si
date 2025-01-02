import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
from datasets import DATASETS_PATH
import numpy as np
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression

class TestLassoRegressor(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        lasso = LassoRegression()
        lasso.fit(self.train_dataset)

        self.assertTrue(isinstance(lasso.fit(self.train_dataset), LassoRegression))
        self.assertEqual(lasso.theta.shape[0], self.train_dataset.X.shape[1])
        self.assertIsNotNone(lasso.theta)
        self.assertIsNotNone(lasso.mean)
        self.assertIsNotNone(lasso.std)

    def test_predict(self):
        lasso = LassoRegression()
        lasso.fit(self.train_dataset)
        predictions = lasso.predict(self.test_dataset)

        self.assertTrue(isinstance(predictions, np.ndarray))
        self.assertEqual(predictions.shape, self.test_dataset.y.shape)
    
    def test_score(self):
        lasso = LassoRegression()
        lasso.fit(self.train_dataset)
        score = lasso.score(self.test_dataset)

        self.assertTrue(isinstance(score, float))
    
    def test_cost(self):
        lasso = LassoRegression()
        lasso.fit(self.train_dataset)
        lasso.predict(self.test_dataset)
        cost = lasso.cost(self.test_dataset)

        self.assertTrue(isinstance(cost, float))

    def test_soft_threshold(self):
        lasso = LassoRegression()
        self.assertEqual(lasso.soft_threshold(3.0, 1.0), 2.0)
        self.assertEqual(lasso.soft_threshold(-3.0, 1.0), -2.0)
        self.assertEqual(lasso.soft_threshold(0.5, 1.0), 0.0)
    
    def test_no_scale(self):
        model_no_scale = LassoRegression(scale=False)
        model_no_scale.fit(self.train_dataset)
        predictions = model_no_scale.predict(self.test_dataset)
        cost = model_no_scale.cost(self.test_dataset)
        score = model_no_scale.score(self.test_dataset)

        self.assertIsNone(model_no_scale.mean)
        self.assertIsNone(model_no_scale.std)
        self.assertIsNotNone(model_no_scale.theta)
        self.assertIsNotNone(predictions)
        self.assertGreaterEqual(score, 0)
        self.assertIsNotNone(cost)

if __name__ == '__main__':
    unittest.main()


