# Why is this even needed?

# a neural network can theoretically be trained on a model that uses scalars, but this is 
# practically impossible 
import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        # gradient is initialised to zero, will be updated during backpropagation, ensure its the same shape
        self.grad = np.zeros_like(self.data) 
        self._backward = lambda: None
        self._prev = set()
        self._op = ''
        # for now, we will assume that all tensors require gradients
        self.requires_grad = True
    
    # element wise addition (basically the same as scalar addition)
    def __add__(self, other):
        # If other is a scalar, we need to convert it to a tensor with the same shape as self.data, 
        # so that the addition can be done element-wise.
        other = other if isinstance(other, Tensor) else Tensor(other)
        new = Tensor(self.data + other.data)
        new._prev = {self, other}
        new._op = '+'

        def _backward():
            self.grad += new.grad
            other.grad += new.grad
        
        new._backward = _backward

        return new
    
    def __neg__(self):
        new = Tensor(-self.data)
        new._prev = {self}
        new._op = 'neg'

        def _backward():
            self.grad += -new.grad

        new._backward = _backward
        return new
    
    # python will try this when the left operand doesn't support the operation, so we need to implement this for cases like 2 + tensor
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return (other) + (-self)
    


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
    
    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other
    
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

    # since neural network losses must become scalars, we need a function that sums 
    # all the elements in the tensor to produce a scalar output
    # axises are not handled here for now, current version is good enough for our use case
    def sum(self):
        new = Tensor(np.sum(self.data))
        new._prev = {self}
        new._op = 'sum'

        def _backward():
            # gradient of sum operation is just a tensor of ones (with same shape as self.data)
            # we multiply by new.grad to account for the chain rule
            self.grad += np.ones_like(self.data) * new.grad
        
        new._backward = _backward

        return new
    
    # Now we need to implement matrix multiplication to 
    # calculate the output of a layer given the input and weights in one operation.


    # TODO:
    # matmul, broadcasting, transpose, reshape, axis handling for sum, gradient checking
