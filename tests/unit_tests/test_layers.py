import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.layers import DenseLayer, Dropout
from si.neural_networks.optimizers import Optimizer

class MockOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        return w - self.learning_rate * grad_loss_w

class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_forward_propagation(self):
        dense_layer = DenseLayer(n_units=30)
        dense_layer.set_input_shape((self.dataset.X.shape[1], ))
        dense_layer.initialize(MockOptimizer(0.001))
        output = dense_layer.forward_propagation(self.dataset.X, training=False)
        self.assertEqual(output.shape[0], self.dataset.X.shape[0])
        self.assertEqual(output.shape[1], 30)


    def test_backward_propagation(self):
        dense_layer = DenseLayer(n_units=30)
        dense_layer.set_input_shape((self.dataset.X.shape[1], ))
        dense_layer.initialize(MockOptimizer(learning_rate=0.001))
        dense_layer.forward_propagation(self.dataset.X, training=True)
        input_error = dense_layer.backward_propagation(output_error=np.random.random((self.dataset.X.shape[0], 30)))
        self.assertEqual(input_error.shape[0], self.dataset.X.shape[0])
        self.assertEqual(input_error.shape[1], 9)

class TestDropoutLayer(unittest.TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=0)

        np.random.seed(0) 
        self.dropout_50 = Dropout(probability=0.5)
        self.input_data = [self.dataset.X, np.array([[1, 2, 3], [4, 5, 6]])]

    def test_probability_between_zero_one(self):
        with self.assertRaises(ValueError):
            Dropout(probability=2)
        with self.assertRaises(ValueError):
            Dropout(probability=-0.1)     

    def test_dropout_forward_propagation(self):   
        for prob in [0, 0.2, 0.5, 0.8]:
            dropout = Dropout(probability=prob)
            for input_data in self.input_data:
                output = dropout.forward_propagation(input_data, training=True)
                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape, input_data.shape)
                self.assertGreater(np.sum(output), 0)

                output = dropout.forward_propagation(input_data, training=False)
                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape, input_data.shape)
                self.assertGreater(np.sum(output), 0)

        output = self.dropout_50.forward_propagation(self.input_data[1], training=True)
        self.assertTrue(np.all((output == 0) | (output == 2) | (output == 4) | (output == 6) | (output == 8) | (output == 12)))
     
    def test_dropout_backward_propagation(self):
        for prob in [0, 0.2, 0.5, 0.8]:
            dropout = Dropout(probability=prob)
            for input_data in self.input_data:
                dropout.forward_propagation(input_data, training=True)
                output_error = np.random.random(input_data.shape)
                input_error = dropout.backward_propagation(output_error)
                self.assertEqual(input_error.shape, output_error.shape)

    def test_dropout_output_shape(self):
        for prob in [0, 0.2, 0.5, 0.8]:
            dropout = Dropout(probability=prob)
            for input_data in self.input_data:
                dropout.forward_propagation(input_data, training=False)
                self.assertEqual(dropout.output_shape(), input_data.shape)

    def test_dropout_parameters(self):
        for prob in [0, 0.2, 0.5, 0.8]:
            dropout = Dropout(probability=prob)
            self.assertEqual(dropout.parameters(), 0)

if __name__ == "__main__":
    unittest.main()