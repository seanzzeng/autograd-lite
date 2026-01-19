import torch
import unittest
from engine.scalar import Value

# Unit tests for the Value class in engine.scalar

class TestScalarOperations(unittest.TestCase):
    def test_addition(self):
        a_torch = torch.tensor(2.0, requires_grad=True)
        b_torch = torch.tensor(3.0, requires_grad=True)
        c_torch = a_torch + b_torch

        a = Value(2.0)
        b = Value(3.0)
        c = a + b

        self.assertAlmostEqual(c.data, c_torch.item(), places=5)

    def test_multiplication(self):
        a_torch = torch.tensor(2.0, requires_grad=True)
        b_torch = torch.tensor(3.0, requires_grad=True)
        c_torch = a_torch * b_torch

        a = Value(2.0)
        b = Value(3.0)
        c = a * b

        self.assertAlmostEqual(c.data, c_torch.item(), places=5)

    def test_tanh(self):
        a_torch = torch.tensor(0.5, requires_grad=True)
        c_torch = torch.tanh(a_torch)

        a = Value(0.5)
        c = a.tanh()

        self.assertAlmostEqual(c.data, c_torch.item(), places=5)

    def test_relu(self):
        a_torch = torch.tensor(-1.0, requires_grad=True)
        c_torch = torch.relu(a_torch)

        a = Value(-1.0)
        c = a.relu()

        self.assertAlmostEqual(c.data, c_torch.item(), places=5)

    def test_backward(self):
        a_torch = torch.tensor(2.0, requires_grad=True)
        b_torch = torch.tensor(3.0, requires_grad=True)
        c_torch = a_torch * b_torch + torch.tanh(a_torch)
        c_torch.backward()

        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a.tanh()
        c.backward()

        self.assertAlmostEqual(a.grad, a_torch.grad.item(), places=5)
        self.assertAlmostEqual(b.grad, b_torch.grad.item(), places=5)

    def test_more_ops(self):
        a_torch = torch.tensor(1.5, requires_grad=True)
        b_torch = torch.tensor(2.5, requires_grad=True)
        c_torch = (a_torch - b_torch) ** 2 + torch.exp(a_torch)
        c_torch.backward()

        a = Value(1.5)
        b = Value(2.5)
        c = (a - b) ** 2 + a.exp()
        c.backward()

        self.assertAlmostEqual(a.grad, a_torch.grad.item(), places=5)
        self.assertAlmostEqual(b.grad, b_torch.grad.item(), places=5)

    def test_mixed(self):
        a_torch = torch.tensor(0.7, requires_grad=True)
        b_torch = torch.tensor(-1.2, requires_grad=True)
        c_torch = torch.relu(a_torch * b_torch + torch.tanh(b_torch))
        c_torch.backward()

        a = Value(0.7)
        b = Value(-1.2)
        d = a * b + b.tanh()
        c = d.relu()
        c.backward()

        self.assertAlmostEqual(a.grad, a_torch.grad.item(), places=5)
        self.assertAlmostEqual(b.grad, b_torch.grad.item(), places=5)

    def test_chain(self):
        a_torch = torch.tensor(0.3, requires_grad=True)
        b_torch = torch.tensor(0.8, requires_grad=True)
        c_torch = torch.relu(a_torch + b_torch * torch.tanh(a_torch))
        c_torch.backward()

        a = Value(0.3)
        b = Value(0.8)
        c = a.relu() + b * a.tanh()
        c.backward()

        self.assertAlmostEqual(a.grad, a_torch.grad.item(), places=5)
        self.assertAlmostEqual(b.grad, b_torch.grad.item(), places=5)

    def test_complex_ops(self):
        a_t = torch.tensor([-4.0], requires_grad=True)
        b_t = torch.tensor([2.0], requires_grad=True)
        c_t = a_t + b_t
        d_t = a_t * b_t + b_t**3
        c_t = c_t + c_t + 1
        c_t = c_t + 1 + c_t + (-a_t)
        d_t = d_t + d_t * 2 + (b_t + a_t).relu()
        d_t = d_t + 3 * d_t + (b_t - a_t).relu()
        e_t = c_t - d_t
        f_t = e_t**2
        f_t.backward()

        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        f.backward()

        self.assertAlmostEqual(f.data, f_t.item(), places=5)
        self.assertAlmostEqual(a.grad, a_t.grad.item(), places=5)
        self.assertAlmostEqual(b.grad, b_t.grad.item(), places=5)

if __name__ == '__main__':
    unittest.main()