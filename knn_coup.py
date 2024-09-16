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


customer_matrix = np.array([
    [1, 1, 1, 0, 0],  # Customer 0
    [1, 1, 1, 1, 0],  # Customer 1
    [0, 0, 1, 1, 0],  # Customer 2
    [1, 0, 1, 1, 0],  # Customer 3
    [0, 1, 1, 1, 1]   # Customer 4
])
new_customer = np.array([[1, 0, 0, 1, 0]])  # Interacted with coupon 0 and 3

k=1
model = knn(k=1)

distances = [model.eud(new_customer[0], other) for other in customer_matrix]
nearest_index = np.argsort(distances)[:k]  # Index of the nearest neighbor
nearest_neighbor = customer_matrix[nearest_index]

recommended_coupons = np.where((nearest_neighbor.sum(axis=0) > 0) & (new_customer == 0))[1]
print(nearest_neighbor)
print(new_customer)
print(recommended_coupons)