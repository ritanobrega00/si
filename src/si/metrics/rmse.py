import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, Y_pred) -> float:
    """
    Aim: Calculate Root Mean Squared Error (RMSE) between the true (observed) and predicted values

    Parameters:
    y_true: np.array -> The true values
    Y_pred: np.array -> The predicted values

    Returns:
    float -> The root mean squared error between the true and predicted values 
    -> Low RMSE: Indicates that the model's predictions are close to the actual values, better performance.
    -> High RMSE: Indicates larger errors in predictions, worse performance.
    """
    return np.sqrt(mean_squared_error(y_true, Y_pred))