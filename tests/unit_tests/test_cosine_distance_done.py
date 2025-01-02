import unittest
import numpy as np

from si.statistics.cosine_distance import cosine_distance
from sklearn.metrics.pairwise import cosine_distances

class TestCosineDistance(unittest.TestCase):
    def setUp(self):
        self.x = [ np.array([1, 2, 3]) , np.array([3, 4, 0]) ]
        self.y = [ np.array([[1, 2, 3], [4, 5, 6]]) , np.array([[0, 0, 1], [1, 1, 1], [3, 4, 0]]) ]
    
    def test_cosine_distance(self):
        for sample in self.x:
            for samples in self.y:
                custom_distance = cosine_distance(sample, samples)
                sklearn_distance = cosine_distances(sample.reshape(1, -1) , samples)
                assert np.allclose(custom_distance, sklearn_distance)

    def test_cosine_distance_with_zeros(self):
        x = np.array([0, 0, 0])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ZeroDivisionError):
            cosine_distance(x, y)

    def test_cosine_distance_with_negative_values(self):
        x = np.array([-1, -2, -3])
        y = np.array([[1, 2, 3], [-4, -5, -6]])
        result = cosine_distance(x, y)
        sklearn_distance = cosine_distances(x.reshape(1, -1), y)
        assert np.allclose(result, sklearn_distance)

if __name__ == '__main__':
    unittest.main()
