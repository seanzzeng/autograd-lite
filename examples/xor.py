# https://dev.to/jbahire/demystifying-the-xor-problem-1blk <-- article to understand more
# about the XOR problem TLDR: only works with MLP
# 2-layer MLP

import numpy as np
from engine.tensor import Tensor

def main():
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]) # inputs
    Y = Tensor([[0], [1], [1], [0]]) # targets

    np.random.seed(42) # random numbers are always the same -- 42 is for convention

    # input layer: 2 neurons (since 2 things to change)
    # hidden layer: 4 neurons
    # output layer: 1 neuron (0 or 1)
    w1 = Tensor(np.random.randn(2, 4) * 0.1)
    b1 = Tensor(np.zeros((1,4)))
    w2 = Tensor(np.random.randn(4, 1) * 0.1)
    b2 = Tensor(np.zeros((1,4)))

    params = [w1, b1, w2, b2]
    learning_rate = 0.05

    print("start training \n")

    for i in range(150):
        intermediate = (X @ w1 + b1).relu()
        pred = intermediate @ w2 + b2 # no relu on the output otherwise negative numbers are gone

        diff = pred - Y
        loss = (diff * diff).sum()

        for p in params:
            # reset gradients each loop
            p.grad = np.zeros_like(p.data)

        # starts backprop
        loss.backward()

        for p in params:
            p.data -= learning_rate * p.grad

        if i % 20 == 0:
            print(f"Iteration {i}: Loss: {loss.data:.4f}")

    print("\ntraining complete\n")
    print("Final Predictions: (target is [0, 1, 1, 0]):\n")

    final_pred = (X @ w1 + b1).relu() @ w2 + b2
    print(np.round(final_pred.data, 3))

if __name__ == "__main__":
    main()



