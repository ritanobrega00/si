import numpy as np
import itertools
from si.model import Model
from si.dataset import Dataset
from si.model_selection.cross_validate import CrossValidator

class GridSearch:
    def setUp(self, seed: int = None):
        
        self.seed = seed

    def grid_search_cv(model: Model, dataset: Dataset, hyperparameter_grid: dict, scoring: callable, 
                       cv: int, seed: int = None) -> dict:
        for hyperparameter in hyperparameter_grid:
            if not hasattr(model, hyperparameter):
                raise ValueError(f"Model does not have hyperparameter {hyperparameter}")
            
        results = {"scores": [], "hyperparameters": []}
        combinations = itertools.product(hyperparameter_grid.values())
        for combination in combinations:
            for param_name, param_value in zip(hyperparameter_grid.keys(), combination):
                setattr(model, param_name, param_value)
            scores = CrossValidator.k_fold_cross_validation(model, dataset, scoring, cv)
            results["scores"].append(np.mean(scores))
            results["hyperparameters"].append(combination)  

        best_score_index = np.argmax(results["scores"])
        results["best_hyperparameters"] = results["hyperparameters"][best_score_index]
        results["best_score"] = np.max(results["scores"])

        return results