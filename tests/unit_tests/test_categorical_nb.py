import unittest
import numpy as np
from si.data.dataset import Dataset
from si.models.categorical_nb import CategoricalNB


class TestCategoricalNB(unittest.TestCase):
    def setUp(self):
        X = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ])
        y = np.array([1, 0, 1, 0])
        self.dataset = Dataset(X, y)

        self.model = CategoricalNB(smothing=1.0)

    def test_fit(self):
        self.model.fit(self.dataset)

        expected_class_prior = np.array([0.5, 0.5])  # Equal distribution of classes
        np.testing.assert_almost_equal(self.model.class_prior, expected_class_prior, decimal=5)

        expected_feature_probs = np.array([[0.25, 0.5 , 0.25],[0.75, 0.5 , 0.75]])
        np.testing.assert_almost_equal(self.model.feature_probs, expected_feature_probs, decimal=5)

    def test_predict(self):
        self.model.fit(self.dataset)
        X_test = np.array([
            [1, 1, 0],
            [0, 0, 1]
        ])
        dataset_test = Dataset(X=X_test, y=None)

        expected_predictions = np.array([1, 0])
        predictions = self.model.predict(dataset_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, expected_predictions.shape)

    def test_score(self):
        self.model.fit(self.dataset)
        self.model.predict(self.dataset)

        expected_accuracy = 1.0  # Model should perfectly classify the training set
        score = self.model.score(self.dataset)
        self.assertAlmostEqual(score, expected_accuracy, places=5)


if __name__ == '__main__':
    unittest.main()
