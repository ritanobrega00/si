import numpy as np
from si.model import Model
from si.dataset import Dataset

class CrossValidator:
    def setUp(self, seed: int = None):
        self.seed = seed

    def k_fold_cross_validation(model: Model, dataset: Dataset, scoring: callable, cv: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        num_samples = dataset.X.shape[0]

        samples_per_fold = num_samples // cv

        scores = []

        indexes = np.array(num_samples)
        np.random.shuffle(indexes)
        for i in range(cv):
            start = samples_per_fold * i
            end = samples_per_fold * (i + 1)

            test_indexes = indexes[start:end]
            train_indexes = np.concatenate([indexes[:start], indexes[end:]])

            train_dataset = Dataset(X = dataset.X[train_indexes], y= dataset.y[train_indexes])
            test_dataset = Dataset(X = dataset.X[test_indexes], y= dataset.y[test_indexes])

            model.fit(train_dataset)
            predictions = model.predict(test_dataset)
            score = scoring(test_dataset.y, predictions)
            scores.append(score)

        return scores

