import numpy as np
class Kmeans:
    def __init__(self, k , max_iter):
        self.k = k
        self.max_iter= max_iter

    def eu_distance(self, a, b ):
        return np.sqrt(np.sum((a-b)**2))
    
    def assign(self, points, centers):
        distances = np.zeros((len(points), self.k))
        for i in range(len(points)):
            for j in range(self.k):
                distances[i][j] = self.eu_distance(points[i], centers[j])
        return np.argmin(distances, axis =1 )

    def update(self, points, labels):
        new_centers = np.zeros((self.k, len(points[0])))
        for i in range(self.k):
            cluster= points[labels==i]
            new_centers[i]= np.mean(cluster, axis= 0)
        return new_centers

    def fit(self, x):
        self.centroids = x[np.random.choice(len(x), self.k, replace=False )]
        for i in range(self.max_iter):
            self.labels = self.assign(x, self.centroids)
            new_center = self.update(x, self.labels)
            #if np.all(new_center== self.centroids):
            if np.allclose(new_center, self.centroids):   
                break
            self.centroids= new_center
    def predict(self, points):
        return self.assign(points, self.centroids)

X = np.random.rand(100, 2)
X = np.vstack((np.random.randn(100, 2) + np.array([0, 0]),
              np.random.randn(100, 2) + np.array([5, 5]),
              np.random.randn(100, 2) + np.array([0, 5])))
  
kmeans= Kmeans(k =3, max_iter=100)
kmeans.fit(X)
labels = kmeans.predict(X)
print("Labels:", labels)
print("Centroids:", kmeans.centroids)
    