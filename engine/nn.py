import numpy as np
from engine.tensor import Tensor

class Module:
    def zero_grad(self):
        # Clears all gradients
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        # will be overridden
        return []  

class Linear(Module):
    def __init__(self, in_features, out_features):
        # Xavier initialisation https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
        self.W = Tensor(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        self.b = Tensor(np.zeros((1, out_features)))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

class Relu(Module):
    # wrap relu in a class so it behaves similarly to linear layer
    def __call__(self, x):
        return x.relu()
    
    def parameters(self):
        return []

# container module
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params