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
Start training 

Iteration 0: Loss: 6.9946
Iteration 10: Loss: 0.8705
Iteration 20: Loss: 0.5595
Iteration 30: Loss: 0.3457
Iteration 40: Loss: 0.1966
Iteration 50: Loss: 0.1033
Iteration 60: Loss: 0.0517
Iteration 70: Loss: 0.0251
Iteration 80: Loss: 0.0120
Iteration 90: Loss: 0.0057

Training complete

Final Predictions: (target is [0, 1, 1, 0]):

[[0.044]
 [0.988]
 [0.978]
 [0.012]]
```
Here, we can observe that the loss value gradually decreases as the number of iterations increases, proving the library is functional.

## Planning
* Implement Adam to replace vanilla gradient descent, resulting in faster and more stable convergence.
* More examples (MNIST).
* Cross-Entropy Loss & Softmax.
* This currently doesn't support tensors with more than 2 dimensions, so N-dimensional matrix operations could be something to look into.
* Layer Normalisation.

