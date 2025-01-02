import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
from datasets import DATASETS_PATH
from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
import numpy as np
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

from si.ensemble.stacking_classifier import StackingClassifier

class TestStackingClassifier(unittest.TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_csv(filename=self.csv_file, label=True, features= True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=5)
        
        self.model1 = KNNClassifier()
        self.model2 = LogisticRegression()
        self.model3 = DecisionTreeClassifier(max_depth=5)
        self.final_model = KNNClassifier()

        self.stack_classifier = StackingClassifier(models = [self.model1, self.model2, self.model3], 
                                                    final_model = self.final_model)

    def test_stacking_classifier_fit(self):
        self.stack_classifier.fit(self.train_dataset)
        self.assertTrue(hasattr(self.stack_classifier.models[0], 'k'))
        self.assertTrue(hasattr(self.stack_classifier.models[1], 'l2_penalty'))
        self.assertEqual(self.stack_classifier.models[2].max_depth, 5)

    def test_stacking_classifier_predict(self):
        self.stack_classifier.fit(self.train_dataset)
        predictions = self.stack_classifier.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])


    def test_stacking_classifier_score(self):
        self.stack_classifier.fit(self.train_dataset)
        self.stack_classifier.predict(self.test_dataset)
        score = self.stack_classifier.score(self.dataset)

        self.assertEqual(round(score, 2), 0.98)

if __name__ == '__main__':
    unittest.main()