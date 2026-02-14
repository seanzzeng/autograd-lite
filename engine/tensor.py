# Why is this even needed?

# A neural network can theoretically be trained on using scalars alone, but in practice 
# this is highly inefficient. Instead, we use tensors, which are multi-dimensional arrays, to 
# represent inputs, weights, biases, and outputs. This allows for batch processing and 
# parallel computations, significantly speeding up the training process.

import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        # Gradient is initialised to zero, will be updated during backpropagation
        self.grad = np.zeros_like(self.data) 
        self._backward = lambda: None
        self._prev = set()
        self._op = ''
        # for now, we will assume that all tensors require gradients
        self.requires_grad = True
    
    # element wise addition (basically the same as scalar addition)
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        new = Tensor(self.data + other.data)
        new._prev = {self, other}
        new._op = '+'

        def _backward():
            self.grad += new.grad
            other.grad += new.grad
        
        new._backward = _backward

        return new
    
    # element wise multiplication (again, exactly the same)
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        new = Tensor(self.data * other.data)
        new._prev = {self, other}
        new._op = '*'

        def _backward():
            self.grad += new.grad * other.data
            other.grad += new.grad * self.data
        
        new._backward = _backward

        return new
    
    def backward(self):
        
        topo = []
        visited = set()

        # Topological sort
        def build_topo(vertex):
            if vertex not in visited:
                visited.add(vertex)
                for child in vertex._prev:
                    build_topo(child)
                topo.append(vertex)
        build_topo(self)

        # set gradient to all 1's since we are doing dL/dL, but shape must be the same as the tensor's data
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()