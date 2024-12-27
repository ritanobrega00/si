<<<<<<< HEAD
import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
=======
from abc import abstractmethod

import numpy as np


class Optimizer:

>>>>>>> 2c596bf371d8771940bca2540623d5db9e1b5cdc
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
<<<<<<< HEAD
    def update(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.0):
=======
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
>>>>>>> 2c596bf371d8771940bca2540623d5db9e1b5cdc
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