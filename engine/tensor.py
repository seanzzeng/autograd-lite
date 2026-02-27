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
        # numpy broadcasts this for us (forward pass only)
        new = Tensor(self.data + other.data)
        new._prev = {self, other}
        new._op = '+'

        def _backward():
            # reverses the broadcasting that numpy automatically did (if it did broadcast)
            grad_self = new.grad
            grad_other = new.grad

            # in a 2D matrix example, sums all the rows if its adding to a scalar (e.g. bias)
            # so they can actually add (sum because they all contribute). generalises to higher
            # dimensions as well (but this is never used in a MLP)
            while grad_self.ndim > self.data.ndim:
                # axis = 0 refers to the outermost layer (so the first number in 
                # a tensor), all this loop does is equate the number of dimensions
                grad_self = grad_self.sum(axis = 0)
            for index, dim in enumerate(self.data.shape):
                # only dimension size that permits broadcasting (1 is pretty much only 
                # used for broadcasting). the function of this loop is to equate the shape itself
                # (since dimensions equating doesn't imply shape matching)
                if dim == 1:
                    grad_self = grad_self.sum(axis = index, keepdims = True)

            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis = 0)
            for index, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis = index, keepdims = True)


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
            grad_self = new.grad * other.data
            grad_other = new.grad * self.data

            # same reverse broadcasting code as __add__()
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis = 0)
            for index, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis = index, keepdims = True)

            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis = 0)
            for index, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis = index, keepdims = True)


            self.grad += grad_self
            other.grad += grad_other
        
        new._backward = _backward

        return new
    
    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other
    
    def backward(self):
        
        topo = []
        visited = set()

        # Topological sort
        def sort_topo(vertex):
            if vertex not in visited:
                visited.add(vertex)
                for child in vertex._prev:
                    sort_topo(child)
                topo.append(vertex)
        sort_topo(self)

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
    
    # this version of matmul would break for 3D tensors and beyond as tranposing reverses all 
    # dimensions which won't work when backpropagating since numpy does matrix multiplication with
    # tensors by using batched matrix multiplication (however this doesn't matter with MLPs)
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        new = Tensor(self.data @ other.data)
        new._prev = {self, other}
        new._op = '@'

        def _backward():
            # should write an explanation for this formula 
            self.grad += new.grad @ other.data.T
            other.data += self.data.T @ new.grad
        
        new._backward = _backward
        return new
    
    # @ symbol notation
    def __matmul__(self, other):
        return self.matmul(other)
    
    def relu(self):
        new = Tensor(np.maximum(0, self.data))
        new._prev = {self}
        new._op = 'relu'

        def _backward():
            # If output > 0, gradient just passes through, else gradient is 0
            self.grad += (new.data > 0) * new.grad
        new._backward = _backward
        
        return new


    # could do in the futrure
    # transpose, reshape, axis handling for sum (GEEKED), advanced optimisers (ADAM /rmsPrOP)
    # softmax & cross-entropy loss
    # algorithmic graph optimisations (prolly not gonna happen)
    # n dimensional broadcasting
