import unittest 

import numpy as np
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.metrics.rmse import rmse
from si.statistics.sigmoid_function import sigmoid_function

class TestMetrics(unittest.TestCase):

    def test_accuracy(self):

        y_true = np.array([0,1,1,1,1,1,0])
        y_pred = np.array([0,1,1,1,1,1,0])

        self.assertTrue(accuracy(y_true, y_pred)==1)

    def test_mse(self):

        y_true = np.array([0.1,1.1,1,1,1,1,0])
        y_pred = np.array([0,1,1.1,1,1,1,0])

        self.assertTrue(round(mse(y_true, y_pred), 3)==0.004)

    def test_rmse(self):
        y_true = np.array([0.1,1.1,1,1,1,1,0])
        y_pred = np.array([0,1,1.1,1,1,1,0])
        self.assertTrue(isinstance(rmse(y_true, y_pred), float))
        self.assertTrue(round(rmse(y_true, y_pred), 3)==0.065) 
        
        #Test for perfect prediction
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(round(rmse(y_true, y_pred), 1)==0.0)
          
        # Test for single identical element
        y_true = np.array([5.0])
        y_pred = np.array([5.0])
        self.assertTrue(round(rmse(y_true, y_pred), 1)==0.0) 

    def test_sigmoid_function(self):

        x = np.array([1.9, 10.4, 75])

        x_sigmoid = sigmoid_function(x)

        self.assertTrue(all(x_sigmoid >= 0))
        self.assertTrue(all(x_sigmoid <= 1))

if __name__ == '__main__':
    unittest.main()