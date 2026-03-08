import random

from engine.scalar import Value

"""
a very primitive feedforward neural network implementation using the Value class 
for automatic differentiation. essentially copied from Karpathy's micrograd implementation, 
for learning purposes.
"""

class Neuron:
    # A single neuron with 'nin' inputs. This uses a tanh activation function.

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
    # a layer of neurons
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    # Collects all the parameters (i.e. weights and biases of the layer)
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    # MLPs (multilayer perceptrons) are a classic type of feedforward neural network that consists
    # of multiple layers. there are other types of neural networks for more specialised use

    def __init__(self, nin, nouts):
        sz = [nin] + nouts # prepends 'nin' to the list
        # constructs layers one by one, first parameter is the input number of neurons, 
        # second parameter is the output number of neurons
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # this is the forward pass of the neural network
        # overwrites the input with the outputs from each layer as the neural network passes 
        # from Input -> Layer 1 -> Layer 2 ->...
        for layer in self.layers:
            x = layer(x)
        return x
    
    # collects all parameters over whole neural network
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    