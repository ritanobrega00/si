import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest 
from datasets import DATASETS_PATH
from si.data.dataset import Dataset
import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification
import sys
import os 

class TestSelectPercentile(unittest.TestCase):
    def setUp(self):
        self.csv_iris = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset_iris = read_csv(filename=self.csv_iris, features=True, label=True)
        self.csv_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_cpu, features=True, label=True)
        self.datasets = [self.dataset_iris, self.dataset_cpu]

        self.select_percentile_20 = SelectPercentile(score_func=f_classification, percentile=20.0)
        self.select_percentile_50 = SelectPercentile(score_func=f_classification, percentile=50)

    def test_fit(self):
        select_percentile = SelectPercentile(score_func = f_classification, percentile = 20.0)
        for dataset in self.datasets:
            select_percentile.fit(dataset)
            self.assertIsNotNone(select_percentile.F)
            self.assertIsNotNone(select_percentile.p)
            self.assertEqual(len(select_percentile.F), len(dataset.features))
            self.assertEqual(len(select_percentile.p), len(dataset.features))

    def test_transform(self):
        # Test a selected percentile of 50% - select 2 out of 4
        for dataset in self.datasets:
            self.select_percentile_50.fit(dataset)
            transformed_dataset = self.select_percentile_50.transform(dataset)
            self.assertEqual(transformed_dataset.X.shape, (dataset.X.shape[0], int(dataset.X.shape[1] // 2)))
            self.assertEqual(len(transformed_dataset.features), int(len(dataset.features) // 2))

        # Test a selected percentile of 20% - select 1 out of 5
        for dataset in self.datasets:
            self.select_percentile_20.fit(dataset)
            transformed_dataset = self.select_percentile_20.transform(dataset)
            self.assertEqual(transformed_dataset.X.shape, (dataset.X.shape[0], int(dataset.X.shape[1] // 5)))
            self.assertEqual(len(transformed_dataset.features), int(len(dataset.features) // 5))

        
    def test_selected_features(self):    
        # Test that the selected features are the ones with the highest F value
        # Controlled dataset with 4 features
        dataset = Dataset(
            X=np.array([[1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, 5, 6],
                        [4, 5, 6, 7]]),
            y=np.array([0, 1, 0, 1]),
            features=["f1", "f2", "f3", "f4"],
            label="y"
        )
        selector = SelectPercentile(percentile=50) # select 2 out of 4
        selector.fit(dataset)
        sorted_indices = np.argsort(selector.F)[-2:]  
        transformed_dataset = selector.transform(dataset)
        selected_indices = [dataset.features.index(f) for f in transformed_dataset.features]

        self.assertEqual(len(transformed_dataset.features), int(len(dataset.features) // 2))
        self.assertListEqual(sorted_indices.tolist(), sorted(selected_indices))

    def test_extreme_cases(self):
        # Test 0% and 100% percentiles and a dataset with only one feature
        for dataset in self.datasets:
            # Test 0% percentile -> None
            selector_0 = SelectPercentile(percentile=0)
            selector_0.fit(dataset)
            with self.assertRaises(ValueError):
                selector_0.transform(dataset)
                
            # Test 100% percentile
            selector_100 = SelectPercentile(percentile=100)
            selector_100.fit(dataset)
            transformed_dataset_100 = selector_100.transform(dataset)
            self.assertEqual(len(transformed_dataset_100.features), len(dataset.features))  # All features selected

        # Test dataset with only one feature
        dataset = Dataset(
            X=np.array([[1], [2], [3], [4]]),
            y=np.array([0, 1, 0, 1]),
            features=["f1"],
            label="y"
        )
        selector = SelectPercentile(percentile=25.0)
        selector.fit(dataset)
        with self.assertRaises(ValueError):
            selector.transform(dataset)


if __name__ == '__main__':
    unittest.main()