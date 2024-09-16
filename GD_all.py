import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    """
    Perform gradient descent using batch, stochastic, or mini-batch methods.

    Parameters:
    X (np.ndarray): Input feature matrix.
    y (np.ndarray): True target values.
    weights (np.ndarray): Initial weights for the model.
    learning_rate (float): Learning rate for gradient descent.
    n_iterations (int): Number of iterations for the gradient descent.
    batch_size (int): Batch size for mini-batch gradient descent.
    method (str): Gradient descent method ('batch', 'stochastic', 'mini_batch').

    Returns:
    np.ndarray: Updated weights after performing gradient descent.
    """
    n = len(y)  # Number of training samples
    
    for _ in range(n_iterations):
        if method == "batch":
            # Batch Gradient Descent
            y_pred = X.dot(weights)
            loss = np.mean((y - y_pred) ** 2)
            error = (y - y_pred)
            gradient = -(2/n) * X.T.dot(error)
            weights -= learning_rate * gradient
        
        elif method == "stochastic":
            # Stochastic Gradient Descent
            for i in range(n):
                y_pred = X[i].dot(weights)
                error = y[i] - y_pred
                gradient = -2 * X[i].T.dot(error)
                weights -= learning_rate * gradient

        elif method == "mini_batch":
            # Mini-Batch Gradient Descent
            indices = np.arange(n)
            np.random.shuffle(indices)  # Shuffle data for mini-batch selection
            X = X[indices]
            y = y[indices]

            for i in range(0, n, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights -= learning_rate * gradient
                
        else:
            raise ValueError("Invalid method specified. Choose 'batch', 'stochastic', or 'mini_batch'.")

    return weights

# Example usage:
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])
weights = np.zeros(X.shape[1])
learning_rate = 0.01
n_iterations = 1000
batch_size = 2

# Batch Gradient Descent
weights_bgd = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
print("Weights after Batch Gradient Descent:", weights_bgd)

# Stochastic Gradient Descent
weights_sgd = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
print("Weights after Stochastic Gradient Descent:", weights_sgd)

# Mini-Batch Gradient Descent
weights_mbgd = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=batch_size, method='mini_batch')
print("Weights after Mini-Batch Gradient Descent:", weights_mbgd)
