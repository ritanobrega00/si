import numpy as np
import itertools
from typing import Callable, Tuple, Dict, Any
from si.model_selection.cross_validate import k_fold_cross_validation
from si.data.dataset import Dataset


def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, Tuple], scoring: Callable = k_fold_cross_validation,
                         cv: int = 5, n_iter: int = 10) -> Dict[str, Any]:
    """
    Aim: Perform a randomized search cross validation on a model.

    Parameters:
    model: The model to cross validate.
    dataset: Dataset object - The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple] - The hyperparameter grid to use.
    scoring: Callable, default=k_fold_cross_validation - The scoring function to use.
    cv: int, default=5 - The number of folds for cross-validation.
    n_iter: int, default=10 - The number of random combinations to test.

    Returns: Dict[str, Any]
    - The results of the randomized search cross-validation. Includes the scores, hyperparameters,
    best hyperparameters, and best score.
    """
    # Check if the hyperparameteres are valid/if they exist in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    results = {'scores': [], 'hyperparameters': []}

    #Select a set of random combination of hyperparameters from all possibilities
    random_combinations = []
    for _ in range(n_iter):
        combination = {parameter: np.random.choice(values) for parameter, values in hyperparameter_grid.items()}
        random_combinations.append(combination)
    
    for combination in random_combinations:
        # Set the parameteres for each random combination
        for parameter, value in combination.items():
            setattr(model, parameter, value)

        # Cross validate the model
        dataset = Dataset(dataset.X, dataset.y)
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)
        results['scores'].append(np.mean(score)) # Save the mean of the scores
        results['hyperparameters'].append(combination) #Save the hyperparameters

    # Find the best score and respective hyperparameters, and add them to the results
    best_idx = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]

    return results


if __name__ == '__main__':
    # Example usage
    from si.models.logistic_regression import LogisticRegression
    from si.data.dataset import Dataset

    # Load and split the dataset (replace with your dataset)
    dataset_ = Dataset.from_random(600, 100, 2)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Define the hyperparameter grid
    parameter_grid = {'l2_penalty': [1, 10],
    'alpha': [0.001, 0.0001],
    'max_iter': [1000, 2000]
    }


    # Run randomized search cross-validation
    results = randomized_search_cv(model=model,
                                   dataset=dataset_,
                                   hyperparameter_grid=parameter_grid,
                                   cv=3,
                                   n_iter=10)

    # Print results
    print("Results:", results)
    print(f"Best hyperparameters: {results['best_hyperparameters']}")
    print(f"Best score: {results['best_score']}")

