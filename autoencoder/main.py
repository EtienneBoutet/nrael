# %%

from matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(400)

# %%
# Préparation des données
def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# %%
x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = prep_pixels(x_train, x_test)

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels)).T
x_test = x_test.reshape((x_test.shape[0], num_pixels)).T


# %%

# Hidden layer #1 : 256
nodes_h1 = 256
w1 = np.random.randn(nodes_h1, x_train.shape[0])
b1 = np.zeros((nodes_h1, 1))

# Hidden layer #2 : 128
nodes_h2 = 128
w2 = np.random.randn(nodes_h2, nodes_h1)
b2 = np.zeros((nodes_h2, 1))

# Hidden layer (Bottleneck) #3 : 256
nodes_h3 = 256
w3 = np.random.randn(nodes_h3, nodes_h2)
b3 = np.zeros((nodes_h3, 1))

# Hidden layer (Bottleneck) #3 : 784
nodes_h4 = 784
w4 = np.random.randn(nodes_h4, nodes_h3)
b4 = np.zeros((nodes_h4, 1))

# %%
def reluDerivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1

    return x

# %%
m = 60000
learning_rate = 1

costs = []
accuracy = []
epochs = 2

for i in range(epochs):
    Z1 = np.matmul(w1, x_train) + b1
    A1 = np.maximum(Z1, 0)
    
    Z2 = np.matmul(w2, A1) + b2
    A2 = np.maximum(Z2, 0)
    
    Z3 = np.matmul(w3, A2) + b3
    A3 = np.maximum(Z3, 0)

    Z4 = np.matmul(w4, A3) + b4
    A4 = np.maximum(Z4, 0)

    cost = mean_squared_error(x_train, A4)
    costs.append(cost)

    # Backpropagation

    dw4 = (2 / m) * np.matmul(
            np.multiply(Z4 - x_train, reluDerivative(Z4)), 
            A3.T
        )

    # Pas certain de ça
    db4 = (2 / m) * np.sum(Z4 - x_train, axis=1, keepdims=True)

    # à trouver
    err = 0

    dw3 = (2 / m) * np.matmul(
            np.multiply(Z3 - x_train, reluDerivative(Z3)), 
            A2.T
        )

    # Weights and biases update
    w4 = w4 - learning_rate * dw4
    b4 = b4 - learning_rate * db4

    # w3 = w3 - learning_rate * dW3
    # b3 = b3 - learning_rate * db3
    # w2 = w2 - learning_rate * dW2
    # b2 = b2 - learning_rate * db2
    # w1 = w1 - learning_rate * dW1
    # b1 = b1 - learning_rate * db1

    print("Epoch", i, "cost: ", cost)


# %%
