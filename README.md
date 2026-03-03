## Overview

This project is a vectorised reverse-mode library written in Python that is able to run automatic differentiation on NumPy arrays. It works by constructing a dynamic computational graph implicitly during the forward pass. When `.backward()` is called on the loss function, the engine performs a topological sort on the graph, so the gradients w.r.t each parameter can be obtained using the multivariable chain rule.



## Current Capabilities
* **Reverse-Mode Autodiff:** Calculates gradients via backpropagation.
* **Vectorised:** Removes scalar bottlenecks through using tensor operations (including matrix multiplications `@`)
* **Broadcasting:** Handles reverse shape broadcasting during backpropagation. This is done by accumulating gradients across dimensions when adding tensors of different shapes during the backward pass. Note the forward broadcasting is natively handled by NumPy.

## Validation
We focus on the XOR problem for now. This will be replaced by MNIST in the future.
[The logic of the XOR problem is not linearly separable, meaning it cannot be solved by a standard single-layer perceptron.](https://dev.to/jbahire/demystifying-the-xor-problem-1blk)  
We need at least a multilayer perceptron (MLP), which requires an autodiff library. The results matching the targets prove the engine is functional.  
Let's run `examples/xor.py`, noting that the parameters are initialised to random values.
```
start training 

Iteration 0: Loss: 2.0376
Iteration 20: Loss: 0.8842
Iteration 40: Loss: 0.6381
Iteration 60: Loss: 0.2681
Iteration 80: Loss: 0.0537
Iteration 100: Loss: 0.0085
Iteration 120: Loss: 0.0017
Iteration 140: Loss: 0.0003
Iteration 160: Loss: 0.0000
Iteration 180: Loss: 0.0000

training complete

Final Predictions: (target is [0, 1, 1, 0]):

[[0.001]
 [1.   ]
 [1.   ]
 [0.   ]]
```
Here, we can observe that the loss value gradually decreases as the number of iterations increases, proving the library is functional.

## Planning
* Implement Adam to replace vanilla SGD, resulting in faster and more stable convergence.
* More examples (MNIST).
* Cross-Entropy Loss & Softmax.
* This currently doesn't support tensors with more than 2 dimensions, so N-dimensional matrix operations could be something to look into.
* Layer Normalisation.

## References
* **[3B1B: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)**: Intro to neural networks with good visualisations
* **[Micrograd](https://github.com/karpathy/micrograd)**: Inspiration for this project and how I learned the basics of an autodiff engine. `scalar.py` is very similar to micrograd as a result.
* **[The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528)**: Introductory paper to matrix calculus in the context of ML (however, this paper does not explain matmul chain rule)

