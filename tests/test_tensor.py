import torch
import unittest
import numpy as np

from engine.tensor import Tensor

class TestTensorOperations(unittest.TestCase):
    def test_creation(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        a = Tensor(data)

        self.assertIsInstance(a.data, np.ndarray, "should be numpy array")
        self.assertEqual(a.data.shape, (2, 2), "shape is incorrect")
        self.assertIsInstance(a.grad, np.ndarray, "grad should be numpy array")
        self.assertEqual(a.grad.shape, (2, 2), "grad should have same shape")

        np.testing.assert_allclose(a.grad, np.zeros((2, 2)))

if __name__ == '__main__':
    unittest.main()

    