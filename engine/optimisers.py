import numpy as np

class Optimiser:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        # learning rate
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

# stochastic gradient descent (gradiesnt descent with individual training data / batches)
class SGD(Optimiser):
    def __init__(self, parameters, lr = 0.01):
        super().__init__(parameters, lr)

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad