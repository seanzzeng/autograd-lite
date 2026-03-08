import numpy as np
from sklearn.datasets import fetch_openml
from engine.tensor import Tensor
import engine.nn as nn
import engine.optimisers as optim
import matplotlib.pyplot as plt

losses = []
accuracies = []

print("Fetching MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')

# Divide X by 255 so the raw values are normalised between 0 and 1 (so gradrients don't explode)
X_raw = mnist.data.to_numpy() / 255.0
Y_raw = mnist.target.to_numpy().astype(int)

num_samples = X_raw.shape[0]
Y_encoded = np.zeros((num_samples, 10))
Y_encoded[np.arange(num_samples), Y_raw] = 1.0

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLu(),
    nn.Linear(128, 10)
)

optimiser = optim.SGD(model.parameters(), lr = 0.1)

batch_size = 128

print("Start training \n")
for i in range(1001):
    batch = np.random.randint(0, num_samples, size=batch_size)

    X_batch = Tensor(X_raw[batch])
    Y_batch = Tensor(Y_encoded[batch])

    pred = model(X_batch)

    loss = ((pred - Y_batch) * (pred - Y_batch)).sum() / batch_size

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    predictions = np.argmax(pred.data, axis = 1)
    targets = np.argmax(Y_batch.data, axis = 1)
    accuracy = np.mean(predictions == targets) * 100

    losses.append(loss.data)
    accuracies.append(accuracy)

    if i % 100 == 0:
        print(f"Iteration {i:4d} | Loss: {loss.data:.4f} | Batch Accuracy: {accuracy:.2f}%")

print("\nTraining complete\n")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color='tab:red')
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies, color='tab:blue')
plt.title("Batch Accuracy (%)")
plt.xlabel("Step")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()