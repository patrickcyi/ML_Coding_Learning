import numpy as np
import math

class GD:
    def __init__(self, iteration, lr , n_features):
        self.iteration= iteration 
        self.lr = lr 
        self.a= [1]*n_features # slope
        self.b =0 # interce
    def fit(self, x, y ):
        n = len(y)
        for i in range(self.iteration):
            y_pred = np.dot( x, self.a) +self.b
            error = y_pred - y
            gradient_a = 2/n* np.dot(x.T, error)
            gradient_b = 2/n* np.sum(error )

            self.a -= self.lr*gradient_a
            self.b -= self.lr*gradient_b
            loss = np.mean(error**2)

    def predict(self, x):
        return np.dot(x, self.a) + self.b 


X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([5, 7, 9])

    # Initialize and train the model
model = GD(iteration=1000, lr=0.01, n_features=X.shape[1])
model.fit(X, y)

    # Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)