import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(unittest.TestCase):

    def setUp(self):
        self.csv_iris = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset_iris = read_csv(filename=self.csv_iris, features=True, label=True)

        self.test_sizes = [0.2, 0.3, 0.5, 0.7, 0.9]

    def test_train_test_split(self):
        for test_size in self.test_sizes:
            train, test = train_test_split(self.dataset_iris, test_size = test_size, random_state=42)
            test_samples_size = int(self.dataset_iris.X.shape[0] * test_size)
            self.assertEqual(test.X.shape, (test_samples_size, self.dataset_iris.X.shape[1]))
            self.assertEqual(train.X.shape, (self.dataset_iris.X.shape[0] - test_samples_size, self.dataset_iris.X.shape[1]))
            shape_X_dataset = (int(train.X.shape[0] + test.X.shape[0]), int(train.X.shape[1]))
            shape_y_dataset = (int(train.y.shape[0] + test.y.shape[0]), )
            self.assertEqual(self.dataset_iris.X.shape, shape_X_dataset)
            self.assertEqual(self.dataset_iris.y.shape, shape_y_dataset)

    def test_stratified_train_test_split(self):
        for test_size in self.test_sizes:
            train, test = stratified_train_test_split(self.dataset_iris, test_size=test_size, random_state=42)
            test_samples_size = int(self.dataset_iris.X.shape[0] * test_size)
            self.assertEqual(test.X.shape, (test_samples_size, self.dataset_iris.X.shape[1]))
            self.assertEqual(train.X.shape, (self.dataset_iris.X.shape[0] - test_samples_size, self.dataset_iris.X.shape[1]))
            shape_X_dataset = (int(train.X.shape[0] + test.X.shape[0]), int(train.X.shape[1]))
            shape_y_dataset = (int(train.y.shape[0] + test.y.shape[0]), )
            self.assertEqual(self.dataset_iris.X.shape, shape_X_dataset)
            self.assertEqual(self.dataset_iris.y.shape, shape_y_dataset)

            # Test that the class distribution is maintained
            unique_classes, class_counts = np.unique(self.dataset_iris.y, return_counts=True)
            for cls in unique_classes:
                train_class_count = np.sum(train.y == cls)
                test_class_count = np.sum(test.y == cls)
                total_class_count = class_counts[unique_classes == cls][0]
                expected_train = len(train.y) / len(self.dataset_iris.y)
                expected_test = len(test.y) / len(self.dataset_iris.y)
                proportion_train = train_class_count / total_class_count
                proportion_test = test_class_count / total_class_count

                self.assertAlmostEqual(proportion_train, expected_train, delta=0.05)
                self.assertAlmostEqual(proportion_test, expected_test,delta=0.05)

    def test_split_random(self):
        train1, test1 = train_test_split(self.dataset_iris, test_size=0.25, random_state=42)
        train2, test2 = train_test_split(self.dataset_iris, test_size=0.25, random_state=42)
        train_diff, test_diff = train_test_split(self.dataset_iris, test_size=0.25, random_state=123)
        self.assertTrue(np.all(train1.X == train2.X))
        self.assertTrue(np.all(test1.X == test2.X))
        self.assertTrue(np.all(train1.y == train2.y))
        self.assertTrue(np.all(test1.y == test2.y))
            
        self.assertFalse(np.all(train1.X == train_diff.X))
        self.assertFalse(np.all(test1.X == test_diff.X))
        self.assertFalse(np.all(train1.y == train_diff.y))
        self.assertFalse(np.all(test1.y == test_diff.y))

        train1, test1 = stratified_train_test_split(self.dataset_iris, test_size=0.25, random_state=42)
        train2, test2 = stratified_train_test_split(self.dataset_iris, test_size=0.25, random_state=42)
        train_diff, test_diff = stratified_train_test_split(self.dataset_iris, test_size=0.25, random_state=123)
        self.assertTrue(np.all(train1.X == train2.X))
        self.assertTrue(np.all(test1.X == test2.X))
        self.assertTrue(np.all(train1.y == train2.y))
        self.assertTrue(np.all(test1.y == test2.y))

        self.assertFalse(np.all(train1.X == train_diff.X))
        self.assertFalse(np.all(test1.X == test_diff.X))

if __name__ == '__main__':
    unittest.main()