import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
import numpy as np
from datasets import DATASETS_PATH
from si.data.dataset import Dataset
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(unittest.TestCase):

    def setUp(self):
        self.csv_iris = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset_iris = read_csv(filename=self.csv_iris, features=True, label=True)
        self.csv_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_cpu, features=True, label=True)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])
        self.dataset_custom = Dataset(X, y, features=["feature1", "feature2"], label="label")

        self.datasets = [self.dataset_iris, self.dataset_custom]

        self.test_sizes = [0.2, 0.3, 0.5, 0.7, 0.9]

    def test_train_test_split(self):
        for dataset in self.datasets:
            for test_size in self.test_sizes:
                train, test = train_test_split(dataset, test_size = test_size, random_state=42)
                test_samples_size = int(dataset.X.shape[0] * test_size)
                self.assertEqual(test.X.shape, (test_samples_size, dataset.X.shape[1]))
                self.assertEqual(train.X.shape, (dataset.X.shape[0] - test_samples_size, dataset.X.shape[1]))
                shape_X_dataset = (int(train.X.shape[0] + test.X.shape[0]), int(train.X.shape[1]))
                shape_y_dataset = (int(train.y.shape[0] + test.y.shape[0]), )
                self.assertEqual(dataset.X.shape, shape_X_dataset)
                self.assertEqual(dataset.y.shape, shape_y_dataset)

    def test_stratified_train_test_split(self):
        for dataset in self.datasets:
            for test_size in self.test_sizes:
                train, test = stratified_train_test_split(dataset, test_size=test_size, random_state=42)
                test_samples_size = int(dataset.X.shape[0] * test_size)
                self.assertEqual(test.X.shape, (test_samples_size, dataset.X.shape[1]))
                self.assertEqual(train.X.shape, (dataset.X.shape[0] - test_samples_size, dataset.X.shape[1]))
                shape_X_dataset = (int(train.X.shape[0] + test.X.shape[0]), int(train.X.shape[1]))
                shape_y_dataset = (int(train.y.shape[0] + test.y.shape[0]), )
                self.assertEqual(dataset.X.shape, shape_X_dataset)
                self.assertEqual(dataset.y.shape, shape_y_dataset)

                # Check if the class distribution is maintained
               # unique_classes, class_counts = np.unique(self.dataset.y, return_counts=True)
               # for cls in unique_classes:
               #     train_class_count = np.sum(train.y == cls)
                #    test_class_count = np.sum(test.y == cls)
                #    total_class_count = class_counts[unique_classes == cls][0]
                #    self.assertAlmostEqual(train_class_count / total_class_count, 0.8, delta=0.1)
                #    self.assertAlmostEqual(test_class_count / total_class_count, 0.2, delta=0.1)

    def test_split_random(self):
        for dataset in self.datasets:
            train1, test1 = train_test_split(dataset, test_size=0.25, random_state=42)
            train2, test2 = train_test_split(dataset, test_size=0.25, random_state=42)
            train_diff, test_diff = train_test_split(dataset, test_size=0.25, random_state=5)
            self.assertTrue(np.all(train1.X == train2.X))
            self.assertTrue(np.all(test1.X == test2.X))
            self.assertTrue(np.all(train1.y == train2.y))
            self.assertTrue(np.all(test1.y == test2.y))
            
            self.assertFalse(np.all(train1.X == train_diff.X))
            self.assertFalse(np.all(test1.X == test_diff.X))
            self.assertFalse(np.all(train1.y == train_diff.y))
            self.assertFalse(np.all(test1.y == test_diff.y))

            train1, test1 = stratified_train_test_split(dataset, test_size=0.25, random_state=42)
            train2, test2 = stratified_train_test_split(dataset, test_size=0.25, random_state=42)
            train_diff, test_diff = stratified_train_test_split(dataset, test_size=0.25, random_state=5)
            self.assertTrue(np.all(train1.X == train2.X))
            self.assertTrue(np.all(test1.X == test2.X))
            self.assertTrue(np.all(train1.y == train2.y))
            self.assertTrue(np.all(test1.y == test2.y))

            self.assertFalse(np.all(train1.X == train_diff.X))
            self.assertFalse(np.all(test1.X == test_diff.X))


if __name__ == '__main__':
    unittest.main()