## Overview

This project is a vectorised reverse-mode library written in Python that is able to run automatic differentiation on NumPy arrays. It works by constructing a dynamic computational graph implicitly during the forward pass. When `.backward()` is called on the loss function, the engine performs a topological sort on the graph, so the gradients w.r.t each parameter can be obtained using the multivariable chain rule.



## Current Capabilities
* **Reverse-Mode Autodiff:** Calculates gradients via backpropagation.
* **Vectorised:** Removes scalar bottlenecks through using matrix operations (including matrix multiplications `@`)
* **Broadcasting:** Handles reverse shape broadcasting during backpropagation. This is done by accumulating gradients across dimensions when adding tensors of different shapes during the backward pass. Note the forward broadcasting is natively handled by NumPy.
* **Abstraction:** Replicates PyTorch's API (e.g. nn.Sequential)

## Demo: XOR
We focus on the XOR problem as it is the simplest application of machine learning.
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

## Demo: MNIST
This engine is capable of training a MLP on the MNIST dataset (100,000+ params) to ~90% accuracy using Mean Squared Error (MSE) loss. 
We can check this by executing `examples/mnist.py`.
```
Fetching MNIST dataset...
Start training 

Iteration    0 | Loss: 1.3746 | Batch Accuracy: 7.03%
Iteration  100 | Loss: 0.2722 | Batch Accuracy: 89.84%
Iteration  200 | Loss: 0.2186 | Batch Accuracy: 92.19%
Iteration  300 | Loss: 0.1848 | Batch Accuracy: 94.53%
Iteration  400 | Loss: 0.1934 | Batch Accuracy: 93.75%
Iteration  500 | Loss: 0.1825 | Batch Accuracy: 93.75%
Iteration  600 | Loss: 0.1626 | Batch Accuracy: 96.09%
Iteration  700 | Loss: 0.1500 | Batch Accuracy: 95.31%
Iteration  800 | Loss: 0.2046 | Batch Accuracy: 90.62%
Iteration  900 | Loss: 0.1746 | Batch Accuracy: 92.19%
Iteration 1000 | Loss: 0.1938 | Batch Accuracy: 93.75%

Training complete
```
We can visualise this using matplotlib:
![MNIST Training Progress](mnist_training_results.png)

## Planning
* Implement Adam to replace stochastic gradient descent, resulting in faster and more stable convergence.
* Cross-Entropy Loss & Softmax.
* This currently doesn't support tensors with more than 2 dimensions, so N-dimensional matrix operations could be something to look into.
* Layer Normalisation.

