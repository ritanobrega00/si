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
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)
        input_data = np.array([-1, 0, 1])
        self.tanh = TanhActivation()

    def test_tanh_activation_function(self):
        
        expected_output = np.tanh(input_data)
        output = self.tanh.activation_function(input_data)
        self.assertTrue(np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}")

    def test_tanh_derivative(self):
        tanh = TanhActivation()
        input_data = np.array([-1, 0, 1])
        expected_derivative = 1 - np.tanh(input_data) ** 2
        derivative = tanh.derivative(input_data)
        self.assertTrue(np.allclose(derivative, expected_derivative), f"Expected {expected_derivative}, but got {derivative}")

class TestSoftmaxActivation(unittest.TestCase):  
    def test_softmax_activation_function(self):
        softmax = SoftmaxActivation()
        input_data = np.array([[1, 2, 3], [1, 2, 3]])
        exp_values = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        expected_output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        output = softmax.activation_function(input_data)
        self.assertTrue(np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}")

    def test_softmax_derivative(self):
        softmax = SoftmaxActivation()
        input_data = np.array([[1, 2, 3], [1, 2, 3]])
        softmax_output = softmax.activation_function(input_data)
        expected_derivative = softmax_output * (1 - softmax_output)
        derivative = softmax.derivative(input_data)
        self.assertTrue(np.allclose(derivative, expected_derivative), f"Expected {expected_derivative}, but got {derivative}")


if __name__ == "__main__":
    unittest.main()