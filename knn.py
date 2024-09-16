import numpy as np
from collections import Counter
import math

class knn:
    def __init__(self, k ):
        self.k = k 

    def eud(self, a, b ):
        return np.sqrt(np.sum((a-b)**2))

    def predict(self, x, y, x_test):
        prediction=[]
        for point in x_test:
            distances=[ self.eud(point, other ) for other in x]
            k_index = np.argsort(distances)[:self.k]
            k_label = [y[i] for i in k_index]

            most_common = Counter(k_label).most_common(1)
            prediction.append(most_common[0][0])
        return prediction
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 5]])
y_train = np.array([0, 0, 1, 1])

    # Sample test data
X_test = np.array([[1, 1], [4, 4]])

    # Predict the classes using k-NN
k = 3
knn_= knn(k=3)

predictions = knn_.predict(X_train, y_train, X_test)
print(predictions)