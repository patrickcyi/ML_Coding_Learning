# ML coding: 推导backprop(apply chain rule)，
# 然后实现一些operator的backprop step，loss function等，
# 最后是测试以及讨论一些numerical stability。

import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Loss function and its derivative
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))  # Adding a small constant to avoid log(0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# Initialize neural network parameters
input_size = 2   # Number of input features
hidden_size = 2  # Number of neurons in the hidden layer
output_size = 2  # Number of output classes
learning_rate = 0.01

# Random initialization of weights and biases
np.random.seed(42)
W1 = np.random.randn(hidden_size, input_size)
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size)
b2 = np.zeros((output_size, 1))

# Training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).T  # XOR Problem

# Training the neural network
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    # Compute loss
    loss = cross_entropy_loss(Y, A2)

    # Backward pass
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Test the neural network
Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = softmax(Z2)

print("\nFinal Output Predictions:")
print(A2)
