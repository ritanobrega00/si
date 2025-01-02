import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import numpy as np

from datasets import DATASETS_PATH 
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.statistics.cosine_distance import cosine_distance
from 

class TestCosineDistance(unittest.TestCase):
    def setUp(self):
        x1 = np.array([1, 2, 3])
        y1 = np.array([[1, 2, 3], [4, 5, 6]])

    
    def test_cosine_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        expected_distance = np.array([0.0, 0.00217768])
        result = cosine_distance(x, y)
        np.testing.assert_almost_equal(result, expected_distance, decimal=5)

    def test_cosine_distance_with_zeros(self):
        x = np.array([0, 0, 0])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ZeroDivisionError):
            cosine_distance(x, y)

    def test_cosine_distance_with_negative_values(self):
        x = np.array([-1, -2, -3])
        y = np.array([[1, 2, 3], [-4, -5, -6]])
        expected_distance = np.array([2.0, 0.00217768])
        result = cosine_distance(x, y)
        np.testing.assert_almost_equal(result, expected_distance, decimal=5)

if __name__ == '__main__':
    unittest.main()
