import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np
from datasets import DATASETS_PATH
from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression


class TestRandomizedSearchCV(unittest.TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset_bb = read_csv(filename=self.csv_file, label=True, features= True)

        self.dataset_custom = Dataset.from_random(600, 100, 2)
        self.model_log = LogisticRegression()
    
        self.parameter_grid_custom = {  'l2_penalty': [1, 10],
                                        'alpha': [0.001, 0.0001],
                                        'max_iter': [1000, 2000]}

        self.parameter_grid = {'l2_penalty': np.linspace(1, 10, 10),
                                'alpha': np.linspace(0.001, 0.0001, 10),
                                'max_iter': np.linspace(1000, 2000, 200)}

    def test_randomized_search_cv(self):
        n_iter = 10
        for dataset in [self.dataset_bb, self.dataset_custom]:
            results = randomized_search_cv(model=self.model_log,
                                           dataset=dataset,
                                           hyperparameter_grid=self.parameter_grid, 
                                           n_iter = n_iter)
            self.assertIn('scores', results)
            self.assertIn('hyperparameters', results)
            self.assertIn('best_hyperparameters', results)
            self.assertIn('best_score', results)
            self.assertIn('l2_penalty', results['best_hyperparameters'])

            self.assertEqual(len(results['scores']), n_iter)
            self.assertEqual(len(results['hyperparameters']), n_iter)
            self.assertIsInstance(results['best_score'], float)
            self.assertIsInstance(results['best_hyperparameters'], dict)
            self.assertEqual(len(results['best_hyperparameters']), 3)

    def test_invalid_hyperparameter(self):
        # Define an invalid hyperparameter grid
        invalid_parameter_grid = {
            'invalid_param': [1, 10]
        }
        for dataset in [self.dataset_bb, self.dataset_custom]:
            with self.assertRaises(AttributeError):
                randomized_search_cv(model=self.model_log,
                                 dataset=dataset,
                                 hyperparameter_grid=invalid_parameter_grid,
                                 cv=3,
                                 n_iter=10)

    def test_zero_iterations(self):
        n_iter = -6
        for dataset in [self.dataset_bb, self.dataset_custom]:
            results = randomized_search_cv(model=self.model_log,
                                       dataset=dataset,
                                       hyperparameter_grid=self.parameter_grid,
                                       n_iter=n_iter)
            self.assertIn('scores', results)
            self.assertIn('hyperparameters', results)
            self.assertIn('best_hyperparameters', results)
            self.assertIn('best_score', results)

            self.assertEqual(len(results['scores']), 0)
            self.assertEqual(len(results['hyperparameters']), 0)

if __name__ == '__main__':
    unittest.main()