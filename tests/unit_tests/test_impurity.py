import unittest
import numpy as np
from si.statistics.impurity import entropy_impurity, gini_impurity

class TestImpurityFunctions(unittest.TestCase):
    def setUp(self):
        self.dataset_pure = np.array([1, 1, 1, 1]) 
        self.dataset_balanced_binary = np.array([1, 0, 1, 0]) 
        self.dataset_multiclass = np.array([1, 0, 2, 1, 2])  

    def test_entropy_impurity(self):
        self.assertIsInstance(entropy_impurity(self.dataset_pure), float)
        self.assertAlmostEqual(entropy_impurity(self.dataset_pure), 0.0, places=5)
        self.assertAlmostEqual(entropy_impurity(self.dataset_balanced_binary), 1.0, places=5)
        self.assertAlmostEqual(entropy_impurity(self.dataset_multiclass),1.52192809489, places=5)

    def test_gini_impurity(self):
        self.assertIsInstance(gini_impurity(self.dataset_pure), float)
        self.assertAlmostEqual(gini_impurity(self.dataset_pure), 0.0, places=5)
        self.assertAlmostEqual(gini_impurity(self.dataset_balanced_binary), 0.5, places=5)
        self.assertAlmostEqual(gini_impurity(self.dataset_multiclass), 0.6399999999999999, places=5)

if __name__ == '__main__':
    unittest.main()