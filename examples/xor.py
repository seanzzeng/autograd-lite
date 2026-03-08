# https://dev.to/jbahire/demystifying-the-xor-problem-1blk <-- article to understand more
# about the XOR problem TLDR: only works with MLP
# 2-layer MLP

import numpy as np
from engine.tensor import Tensor
import engine.nn as nn
import engine.optimisers as optim

def main():
    np.random.seed(67)

    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]) # inputs
    Y = Tensor([[0], [1], [1], [0]]) # targets

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLu(),
        nn.Linear(16, 1)
    )

    learning_rate = 0.025

    # input layer: 2 neurons (since 2 things to change)
    # hidden layer: 16 neurons
    # output layer: 1 neuron (0 or 1)

    optimiser = optim.SGD(model.parameters(), lr = learning_rate)

    print("Start training \n")

    for i in range(100):
        pred = model(X)

        # mean squared error for loss
        loss = ((pred - Y) * (pred - Y)).sum()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if i % 10 == 0:
            print(f"Iteration {i}: Loss: {loss.data:.4f}")

    print("\nTraining complete\n")
    print("Final Predictions: (target is [0, 1, 1, 0]):\n")

    final_pred = model(X)
    print(np.round(final_pred.data, 3))

if __name__ == "__main__":
    main()



