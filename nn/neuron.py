import random

from engine.scalar import Value

"""
A very primitive feedforward neural network implementation using the Value class 
for automatic differentiation. Essentially copied from Karpathy's micrograd implementation, 
for learning purposes.
"""

class Neuron:
    """
    A single neuron with 'nin' inputs. This uses a tanh activation function.
    """
    def __init__(self, nin):
        # nin = Number of inputs, basically children of this node
        # Initialises all weights and the bias with random numbers between -1 and 1 
        # (might be related to the activation function being tanh?)
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        activation = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    """A layer of neurons."""
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    # Collects all the parameters (i.e. weights and biases of the layer)
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    """A multi-layer perceptron (MLP) neural network. 
    
    The MLP applies a series of layers to the input data to produce an output."""
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # Prepends 'nin' to the list
        # Constructs layers one by one, first parameter is the input number of neurons, 
        # second parameter is the output number of neurons
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # This is the forward pass of the neural network
        # Overwrites the input with the outputs from each layer as the neural network passes 
        # from Input -> Layer 1 -> Layer 2 ->...
        for layer in self.layers:
            x = layer(x)
        return x
    
    # Collects all parameters over whole neural network
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    