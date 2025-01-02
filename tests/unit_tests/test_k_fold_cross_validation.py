import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
from datasets import DATASETS_PATH

from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.cross_validate import k_fold_cross_validation
from si.models.logistic_regression import LogisticRegression

import numpy as np
class TestKFoldCrossValidation(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_k_fold_cross_validation(self):

        model = LogisticRegression()

        scores = np.array(k_fold_cross_validation(model, self.dataset, scoring=accuracy, cv=5))

        self.assertEqual(np.round(np.mean(scores), 2), 0.97)
        self.assertAlmostEqual(np.round(np.std(scores), 2), 0.01, delta=0.1)

if __name__ == '__main__':
    unittest.main() 