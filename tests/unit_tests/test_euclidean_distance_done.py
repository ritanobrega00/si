import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.statistics.euclidean_distance import euclidean_distance

class TestEuclideanDistance(unittest.TestCase):

    def test_euclidean_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        our_distance = euclidean_distance(x, y)
        from sklearn.metrics.pairwise import euclidean_distances
        sklearn_distance = euclidean_distances(x.reshape(1, -1), y)
        assert np.allclose(our_distance, sklearn_distance)

if __name__ == '__main__':
    unittest.main()