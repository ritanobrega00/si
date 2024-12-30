import numpy as np
from typing import List
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):
    """
    Stacking Classifier.

    Parameters:
    - models: List[Model] – List of base models to train the ensemble.
    - final_model: Model – Model to make the final predictions based on the predictions of the base models.

    Methods:
    - _fit: train the ensemble models
    - _predict: predicts the labels using the ensemble models
    - _score: computes the accuracy between predicted and real labels
    """
    def __init__(self, models: List[Model], final_model: Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model
        self.base_predictions = [] # to store the predictions that will be used in the methods

    def _fit(self,dataset: Dataset) -> 'StackingClassifier':
        """
        Aim: Train the base models and the final model using the predictions of the base models.
        Parameters: dataset: Dataset - The training data.
        Returns: self: StackingClassifier - The fitted model.
        """
        self.base_predictions = []
        # Train the initial set of models
        for modelo in self.models:
            modelo.fit(dataset)
            self.base_predictions.append(modelo.predict(dataset))
        
        # Convert base predictions to a dataset for the final model
        self.base_predictions = np.column_stack(self.base_predictions)
        final_dataset = Dataset(self.base_predictions, dataset.y)

        # Train the final model using the predictions of base models
        self.final_model.fit(final_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Aim: Predict the labels using the ensemble models (base and final models).
        Parameters: dataset: Dataset - The test data.
        Returns: np.ndarray - The predicted labels.
        """
        final_predictions = self.final_model.predict(Dataset(self.base_predictions, dataset.y)) 
        return final_predictions

    def _score(self, dataset: Dataset) -> float:
        """ 
        Aim: Compute the accuracy between predicted and real labels.
        Parameters: dataset: Dataset - The test data.
        Returns: float - The accuracy of the model.
        """

        predictions = self.predict(dataset)
        score = accuracy(dataset.y, predictions)
        return score
