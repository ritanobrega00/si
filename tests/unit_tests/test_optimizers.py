import unittest
import numpy as np
from si.neural_networks.optimizers import Optimizer, SGD, Adam

class TestOptimizers(unittest.TestCase):

    def test_sgd_update(self):
        # Initialize optimizer
        sgd = SGD(learning_rate=0.1, momentum=0.9)

        # Define weights and gradients
        w = np.array([0.5, -0.3, 1.0])
        grad_loss_w = np.array([0.1, -0.2, 0.3])

        # Perform update
        updated_w = sgd.update(w, grad_loss_w)

        # Expected retained gradient
        expected_retained_gradient = 0.9 * np.zeros_like(w) + 0.1 * grad_loss_w

        # Expected updated weights
        expected_updated_w = w - 0.1 * expected_retained_gradient

        # Assert retained gradient and updated weights
        np.testing.assert_almost_equal(sgd.retained_gradient, expected_retained_gradient, decimal=6)
        np.testing.assert_almost_equal(updated_w, expected_updated_w, decimal=6)

    def test_adam_update(self):
        # Initialize optimizer
        adam = Adam(learning_rate=0.01)

        # Define weights and gradients
        w = np.array([0.5, -0.3, 1.0])
        grad_loss_w = np.array([0.1, -0.2, 0.3])

        # Perform update
        updated_w = adam.update(w, grad_loss_w)

        # Manually compute expected m, v, m_hat, v_hat, and updated weights
        beta_1, beta_2 = 0.9, 0.999
        epsilon = 1e-8

        # m and v after one update
        m = beta_1 * np.zeros_like(w) + (1 - beta_1) * grad_loss_w
        v = beta_2 * np.zeros_like(w) + (1 - beta_2) * (grad_loss_w ** 2)

        # Bias-corrected estimates
        m_hat = m / (1 - beta_1)
        v_hat = v

if __name__ == '__main__':
    unittest.main()