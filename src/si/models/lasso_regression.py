import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegression(Model):
    """
    Lasso Regression using coordinate descent to optimize the L1-regularized cost function.

    parameters:
    l1_penalty: float, default=1.0 - L1 regularization parameter (lambda) - the higher the value, the more sparsity
    scale: bool, default=True - wheter to scale the data or not
    max_iter: int, default=1000 - maximum number of iterations
    patience: int, default=5 - number of iterations without improvement before stopping the training

    estimated parameters:
    theta: np-array - the coefficients of the model for every feature
    theta_zero: float -the zero coefficient (y intercept)
    mean -mean of the dataset (for every feature)
    std -standard deviation of the dataset (for every feature)

    methods:
    _fit -estimates the theta and theta_zero coefficients, mean and std, using corrdinate descent
    _predict -predicts the dependent variable (y) using the estimated theta coefficients
    _score -calculates the error between the real and predicted y values
    """

    def __init__(self, l1_penalty: float = 1.0, scale: bool = True, max_iter: int = 1000, patience: int = 5, **kwargs):
        super().__init__(**kwargs)
        #Parameters
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        #Attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def soft_threshold(self, rho: float, l1_penalty: float) -> float:
        """
        Aim: computes the soft threshold value
        
        Parameters:
        rho: float -> the value to apply the threshold
        l1_penalty: float -> the threshold value
        
        Returns:
        float -> the thresholded value
        """
        if rho < -l1_penalty:
            return rho + l1_penalty
        elif rho > l1_penalty:
            return rho - l1_penalty
        else:
            return 0
    
    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Aim: estimates the theta and theta_zero coefficients, mean and std, using corrdinate descent
        
        Parameters:
        dataset: Dataset object -> The dataset to fit the model

        Returns:
        self: LassoRegression (the fitted model)
        """

        # Scale the data if not sclaled yet (when scale=True) -> (X - mean) / std
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        
        m, n_features = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n_features)
        self.theta_zero = 0

        # Apply coordinate descent algorithm 
        # - an optimization algorithm that updates one coefficient at a time while keeping the others fixed
        i = 0 
        early_stopping = 0
        # Iterate for max_iter times or until patience is reached
        while i < self.max_iter and early_stopping < self.patience:
            #predict y
            y_pred = np.dot(X, self.theta) + self.theta_zero
            
            #update theta zero
            self.theta_zero = np.mean(dataset.y - np.dot(X, self.theta))

            # Update coefficients
            for j in range(n_features):
                # Compute the residuals excluding current feature's contribution
                residuals = dataset.y - (self.theta_zero + np.dot(X, self.theta) - X[:, j] * self.theta[j])
                rho = np.dot(X[:, j], residuals)

                # Update theta_j using soft-thresholding
                self.theta[j] = self.soft_threshold(rho, self.l1_penalty) / np.sum(X[:, j]**2)

            # Compute the cost
            self.cost_history[i] = self.cost(dataset)

            #Check early stopping
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        #Scale the data and the use the estimations from the fit method
        if self.scale:
            X = (dataset.X - self.mean) / self.std 
        else:
            X = dataset.X
        # Predict y using the estimated theta and theta_zero and X.dot(theta) for matriz multiplication
        y_pred = X.dot(self.theta) + self.theta_zero
        return y_pred
    
    def _score(self, dataset: Dataset) -> float:
        #Predict y using the predict method
        y_pred = self._predict(dataset)
        #Calculate the mse score (error between the predicted and real y values)
        score = mse(dataset.y, y_pred)
        return score
    
    if __name__ == '__main__':
        from si.data.dataset import Dataset
        from si.models.lasso_regression import LassoRegression

        # Linear dataset
        x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(x, np.array([1, 2])) + 3
        dataset = Dataset(X=x, y=y)

        # fit the model
        model = LassoRegression(l1_penalty=1.0, scale=True)
        model.fit(dataset)

        # get coefs
        print(f"Parameters: {model.theta}")

        # compute the score
        score = model.score(dataset)
        print(f"Score: {score}")

        # predict
        y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
        print(f"Predictions: {y_pred_}")