import numpy as np
class Batch:
    def __init__(self, features):
        m, n  = features.shape  # m = number of samples, n = number of features
        print(m, n)
        self.W1 = np.ones((n, n))  # Weight matrix shape should match (n, n) for dot product with (n,)
        self.b1 = np.ones(n)  # Define b1 instead of b2, since you're using b1 in forward pass
        self.gamma = np.ones(n)
        self.beta = np.ones(n)
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def batch_norm(Z, gamma, beta):
        batch_mean  = np.mean(x, axis =0)
        batch_var = np.var(x, axis = 0)

        x_norm = (x- batch_mean)/ np.sqrt(batch_var +1e-5)
        x_out = self.gamma * x_norm + self.beta
        # if training 
        self.prev_mean += batch_mean 
        self.prev_var += batch_var

        return x_out


# Forward pass without batch normalization
def forward_pass(X):
    Z1 = np.dot(W1, X) + b1
    Z1_norm = batch_norm(Z1, gamma, beta)

    A1 = sigmoid(Z1_norm)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2

# Example input
X = np.array([[1, 2], [2, 3]])  # 2 samples, 2 features each

output = forward_pass(X)
print("Output without Batch Normalization:", output)

