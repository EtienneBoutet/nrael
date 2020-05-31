from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import math
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
        self.points = []
        self.X_train = None
        self.y_train = None

    def _euclidian_distance(self, q, p):
        return np.sqrt(np.sum((q - p)**2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [self._euclidian_distance(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        return Counter(k_nearest_labels).most_common(1)[0][0]


iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNN(k=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)