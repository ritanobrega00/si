import pytest
import numpy as np
from si.model_selection.cross_validate import CrossValidator
from si.model import Model
from si.dataset import Dataset

class DummyModel(Model):
    def fit(self, dataset):
        pass

    def predict(self, dataset):
        return np.zeros(dataset.y.shape)

def dummy_scoring(y_true, y_pred):
    return np.mean(y_true == y_pred)

@pytest.fixture
def dataset():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return Dataset(X, y)

@pytest.fixture
def model():
    return DummyModel()

def test_k_fold_cross_validation(dataset, model):
    cv = 5
    scores = CrossValidator.k_fold_cross_validation(model, dataset, dummy_scoring, cv, seed=42)
    assert len(scores) == cv
    assert all(isinstance(score, float) for score in scores)