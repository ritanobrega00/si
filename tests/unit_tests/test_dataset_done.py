import unittest

import numpy as np
import pandas as pd

from si.data.dataset import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        X = np.array([[np.nan, 3.0, 5.0, np.nan], [4, 3, 5.6, 5], [2, 6.6, 6.5, 5]])
        Y = np.array(['no', 'yes', 'yes'])
        features = np.array(['1º', '2º', '3º', '4º'])
        
        self.dataset_nan = Dataset(X, Y, features, label=Y)
        self.original_dataset = Dataset(X, Y, features, label=Y)
        
        
    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dropna(self):
        self.dataset_nan.dropna()
        #original com NaN
        self.assertTrue(bool(np.isnan(self.original_dataset.X).any()))
        self.assertEqual((3,4), self.original_dataset.shape())
        self.assertIn('no', self.original_dataset.y)
        # sem NaN - depois do dropna
        self.assertFalse(bool(np.isnan(self.dataset_nan.X).any()))
        self.assertEqual((2,4), self.dataset_nan.shape())
        self.assertNotIn('no', self.dataset_nan.y)

    def test_fillna(self):
        fill_mean = self.original_dataset.fillna('mean')
        fill_median = self.dataset_nan.fillna('median')

        x = np.array([[np.nan, np.nan, 3], [4, 5, 6]])
        y = np.array([1, 2])
        dataset = Dataset(x, y, features = np.array(['a', 'b', 'c']), label = 'y')
        fill_number = dataset.fillna(111.4)

        fill_mean_dataset = Dataset(X=np.array([[3.,   3.0, 5.0, 5.],   [4., 3, 5.6, 5.], [2., 6.6, 6.5, 5.]]),
                                    y = self.original_dataset.y, 
                                    features=self.original_dataset.features, label=self.original_dataset.label)
        fill_median_dataset = Dataset(X=np.array([[3.,  3.0, 5.0, 5.],  [4, 3, 5.6, 5], [2, 6.6, 6.5, 5]]),
                                    y = self.dataset_nan.y, 
                                    features=self.dataset_nan.features, label=self.dataset_nan.label)
        fill_number_dataset = Dataset(X=np.array([[111.4, 111.4, 3.], [4., 5., 6.]]),
                                    y = dataset.y, features=dataset.features, label=dataset.label)     

        for fill in [fill_mean, fill_median, fill_number]:
            self.assertFalse(bool(np.isnan(fill.X).any()))
        np.testing.assert_array_equal(fill_mean.X, fill_mean_dataset.X)
        np.testing.assert_array_equal(fill_median.X, fill_median_dataset.X)
        np.testing.assert_array_equal(fill_number.X, fill_number_dataset.X)

        with self.assertRaises(ValueError) as context:
            self.original_dataset.fillna('something_invalid')  

    def test_remove_by_index(self):
        self.dataset_nan.remove_by_index([-1])
        self.assertEqual((2,4), self.dataset_nan.shape())
        new_x = np.array([[np.nan, 3.0, 5.0, np.nan], [4, 3, 5.6, 5]])
        np.testing.assert_array_equal(self.dataset_nan.X, new_x)
        
        self.dataset_nan.remove_by_index([0])
        self.assertEqual((1,4), self.dataset_nan.shape())
        self.assertNotIn('no', self.dataset_nan.y)

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_from_to_dataset(self):
        panda_df = self.original_dataset.to_dataframe()
        self.assertTrue(isinstance(panda_df, pd.DataFrame))

        df_again = Dataset.from_dataframe(df = panda_df)
        self.assertTrue(isinstance(df_again, Dataset))
        

if __name__ == '__main__':
    unittest.main()