import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError, CategoricalCrossEntropy


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_categorical_cross_entropy_loss(self):

        error = CategoricalCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_categorical_cross_entropy_derivative(self):

        derivative_error = CategoricalCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])    

if __name__ == "__main__":
    unittest.main()