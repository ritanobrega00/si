import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
from datasets import DATASETS_PATH
import numpy as np
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.activation import ReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation  
class TestSigmoidLayer(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))

    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

class TestTahn(unittest.TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.tanh = TanhActivation()

    def test_tanh_activation_function(self):
        result = self.tanh.activation_function(self.dataset.X)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.all((result >= -1) & (result <= 1))) # all values are between -1 and 1

    def test_tanh_derivative(self):
        derivative = self.tanh.derivative(self.dataset.X)
        self.assertIsInstance(derivative, np.ndarray)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestSoftmaxActivation(unittest.TestCase):  
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.soft = SoftmaxActivation()

    def test_softmax_activation_function(self):
        result = self.soft.activation_function(self.dataset.X)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.all((result >= 0) & (result <= 1))) # all values are between 0 and 1
        self.assertTrue(np.allclose(np.sum(result, axis=-1), 1)) # sum of all values in a row is 1

    def test_softmax_derivative(self):
        derivative = self.soft.derivative(self.dataset.X)
        self.assertIsInstance(derivative, np.ndarray)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])

if __name__ == "__main__":
    unittest.main()