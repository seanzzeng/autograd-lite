import math

class Value:
    """
    This class stores a scalar value and its gradient. It supports very basic arithmetic operations.
    Essentially copied from Karpathy's micrograd implementation, for learning purposes.
    """
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data

        self._prev = set(_children)
        self._op = _op
        self.label = label

        self._backward = lambda: None
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # Wrapping 'other' in the Value class if 'other' is a number (assuming that 'other' is one
        # of the two)
        other = other if isinstance(other, Value) else Value(other)
        new = Value(self.data + other.data, (self, other), '+')

        # Derivatives of child nodes for addition is the same as parent node
        # '+=' here so if a node is used multiple times, the 'contributions' to its derivative are 
        # counted correctly
        def _backward():
            self.grad += 1.0 * new.grad
            other.grad +=  1.0 * new.grad

        new._backward = _backward

        return new
    
    def __radd__(self, other):
        return self + other
    
    def __mul__ (self, other):
        other = other if isinstance(other, Value) else Value(other)
        new = Value(self.data * other.data, (self, other), '*')

        # The backwards function here is basically filling out what operations should be done to 
        # find the derivative of the child nodes with respect to the loss function
        # in the forward pass. Then, the backwards pass actually provides the new.grad value. 
        def _backward():
            self.grad += other.data * new.grad
            other.grad += self.data * new.grad
        new._backward = _backward

        return new
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (other) + (-self)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        new = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * new.grad
        new._backward = _backward

        return new
    
    def exp(self):
        x = self.data
        new = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += new.data * new.grad
        new._backward = _backward

        return new
    
    # activation function for this walkthrough
    # The purpose of activation functions is to introduce some non-linearity into the network, 
    # otherwise the final expression would just be another linear transformation (if you did the math).
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        new = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * new.grad
        new._backward = _backward

        return new
    
    def relu(self):
        new = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')
        def _backward():
            # If output > 0, gradient just passes through, else gradient is 0
            self.grad += (new.data > 0) * new.grad
        new._backward = _backward
        
        return new
    
    def backward(self):
        # topologically orders the graph (any directed edge (u,v), u comes before v)
        # performs backpropagation to compute gradients for all nodes by calling each node's 
        # stored backward function in reverse order.
    
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

        # Set to 1.0 since dL/dL = 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()