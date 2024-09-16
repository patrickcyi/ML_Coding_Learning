import numpy as np
from sklearn.metrics import accuracy_score

class KCV:
    def __init__(self, k , model):
        self.k = k 
        self.model = model 
        self.scores =[]
    def split(self, data , y ):
        index = np.arange(len(data))
        np.random.shuffle(index)
        data= data[index]
        y=y[index]

        fold_size = len(data)//self.k 
        folds =[]

        for i in range(self.k):
            start = i * fold_size
            end = start + fold_size if i < self.k-1 else len(data)
            test = data[start:end]
            train = np.concatenate(data[:start], data[end:])
            
            y_test = y[start:end]
            y_train = np.concatenate(y[:start], y[end:])            
            
            folds.append((train, test, y_train, y_test )    )
        return folds

    def evaluate(self, data, y ):
        folds= self.split(data, y )
        for x_train, x_test, y_train, y_test in folds:
            self.model.fit(x_train, y_train)
            prediction = self.model.predict(x_test)
            score = accuracy_score(y_test, prediction)
            self.scores.append(score)
        return np.mean(self.scores)
   




# Example usage:
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Flatten y for LinearRegression
y = y.flatten()

# Create an instance of the Linear Regression model
model = LinearRegression()

# Create an instance of the KFold class
validator = kfold(model, num_fold=5)

# Evaluate the model
validator.evaluate(X, y)

print("MSE for each fold: ", validator.get_score())