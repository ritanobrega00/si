from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    
class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, 
                 beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1: float
            default = 0.9 - The exponential decay rate for the 1st moment estimates.
        beta_2: float
            default = 0.999 - The exponential decay rate for the 2nd moment estimates.
        epsilon: float
            default = 1e-8 - A small constant for numerical stability.

        Estimated Parameters
        --------------------
        m: numpy.ndarray
            The first moment vector (mean of gradients).
        v: numpy.ndarray
            The second moment vector (mean of squared gradients).
        t: int
            Time step (epoch). - initialized to 0
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None  
        self.v = None  
        self.t = 0     

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Aim: Compute and Update t, m, v and the weights of the layer using the Adam optimization algorithm.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.

        Algorithm:
        m helps smooth the gradients (momentum), making updates more stable. 
        v scales the learning rate based on the magnitude of the gradients, preventing large or unstable updates.
        Bias correction ensures that the moving averages m and v are accurate, especially 
    during the early stages of training when t is small.
        """
        # Check if m and v are initialized, if not initialize them as matrices of zeros
        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        # Increment time step (update t)
        self.t += 1

        # Compute and update m according to the Adam formula
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # Compute and update v according to the Adam formula
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        # Compute m_hat - removes the bias introduced in m during the early stages of training when t is small
        m_hat = self.m / (1 - self.beta_1 ** self.t)

        # Compute v_hat - removes the bias introduced in v during the early stages of training
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Update weights using Adam formula
        w_updated = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w_updated 