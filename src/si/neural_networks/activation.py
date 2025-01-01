from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)

class TanhActivation(ActivationLayer):
    """
    Tanh activation function - applies the tanh function to its input
    It is often used in neural networks to squash the values into a range between -1 and 1
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Aim: applies the Tanh activation function on the input

        Parameters: input - a numpy.ndarray - The input to the layer

        Returns: a numpy.ndarray - The output of the layer
        """
        outputs = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        #As the values must be between -1 and 1 and sum to 1, it will be added a checking to ensure that
        if np.all(output >= -1 and output <= 1 for output in outputs):
            return outputs

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Aim: Derivative of the Tanh activation function.

        Parameters: input - a numpy.ndarray - The input to the layer.

        Returns: a numpy.ndarray - derivate = 1- self.activation_function(input) ** 2
        """
        derivative = 1- self.activation_function(input) ** 2
        return derivative
    
class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function - transforms the raw output scores into a probability distribution 
    (that sums to 1), making it suitable for multi-class classification problems
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters: input - a numpy.ndarray - The input to the layer.

        Returns: a numpy.ndarray - The output of the layer - range between 0 and 1
        """
        # axis=-1 because we want to apply the functions to the last dimension
        # keepdims=True to keep the dimensions of the input
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))
        outputs = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        #As the values must be between 0 and 1 and sum to 1, it will be added a checking to ensure that
        if sum(outputs) == 1 and np.all(output >= 0 and output <= 1 for output in outputs):
            return outputs
        else:
            raise ValueError("Something is wrong: The output values must be between 0 and 1 and sum to 1")

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the Softmax activation function.

        Parameters: input - a numpy.ndarray - The input to the layer.

        Returns: a numpy.ndarray - The derivative = self.activation_function(input) * (1 - self.activation_function(input))
        """
        derivative = self.activation_function(input) * (1 - self.activation_function(input))
        return derivative
