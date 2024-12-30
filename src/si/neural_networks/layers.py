from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

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
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
    
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) 
    
class Dropout(Layer):
    """
    Dropout layer of a neural network.

    Parameters:
    probability: float – the dropout rate, between 0 and 1

    Estimated parameters:
    mask – binomial mask that sets some inputs to 0 based on the probability
    input - the input to the layer
    output - the output of the layer

    Methods:
    forward_propagation – performs forward propagation on the given input
    backward_propagation – performs backward propagation on the given error
    output_shape – returns the input_shape (dropout does not change the shape of the data)
    parameters – returns 0 (droupout layers don't have learnable parameters)

    """

    def __init__(self, probability: float):
        """
        Initialize the Dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, between 0 and 1.
        """
        super().__init__()
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Aim: Perform forward propagation on the given input.

        Parameters:
        input: numpy.ndarray - The input to the layer
        training: bool - to indicate whether the layer is in training mode or in inference mode.

        Returns: a numpy.ndarray - The output of the layer
        """
        self.input = input
        #when in training mode
        if training:
            # Compute the scaling factor
            scaling_factor = 1 / (1 - self.probability)
            # Create a mask using a binomial distribution
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            # Apply the mask and scale the input
            self.output = input * self.mask * scaling_factor
        
        else: # When in inference mode -> return the input
            self.output = input
        
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Aim: Perform backward propagation on the given output error.

        Parameters:
        output_error: numpy.ndarray - The output error of the layer.

        Returns: a numpy.ndarray - The input error of the layer, which is the output error multiplied by the mask.
        """
        return output_error * self.mask

    def output_shape(self) -> tuple:
        return self.input_shape()

    def parameters(self) -> int:
        return 0
