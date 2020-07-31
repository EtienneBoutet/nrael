# %%

import matplotlib
from matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

import numpy as np

np.set_printoptions(suppress=True)
np.random.seed(400)

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

# Entry data setup
x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = prep_pixels(x_train, x_test)

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels)).T
x_test = x_test.reshape((x_test.shape[0], num_pixels)).T

# %%
print(x_train.shape)
# %%

# Weights, bias and neurons creation (3 bottleneck hidden layers)

nodes_h1 = 256
w1 = np.random.randn(nodes_h1, x_train.shape[0])
b1 = np.zeros((nodes_h1, 1))

nodes_h2 = 128
w2 = np.random.randn(nodes_h2, nodes_h1)
b2 = np.zeros((nodes_h2, 1))

nodes_h3 = 256
w3 = np.random.randn(nodes_h3, nodes_h2)
b3 = np.zeros((nodes_h3, 1))

nodes_h4 = 784
w4 = np.random.randn(nodes_h4, nodes_h3)
b4 = np.zeros((nodes_h4, 1))

# %%

# Training setup

def reluDerivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1

    return x

def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

m = 60000
learning_rate = 0.1

costs = []
accuracy = []
epochs = 1000

# %%

# Training
for i in range(epochs):
    Z1 = np.matmul(w1, x_train) + b1
    A1 = np.maximum(Z1, 0)
    
    Z2 = np.matmul(w2, A1) + b2
    A2 = np.maximum(Z2, 0)
    
    Z3 = np.matmul(w3, A2) + b3
    A3 = np.maximum(Z3, 0)

    Z4 = np.matmul(w4, A3) + b4
    A4 = sigmoid(Z4)

    cost = mean_squared_error(x_train, A4)
    costs.append(cost)

    # Back propagation

    dZ4 = A4 - x_train
    # delta = np.ones((A3.shape[0], A3.shape[1]))
    # delta3 = np.multiply(A3, np.subtract(delta, A3))
    
    # bruh = np.matmul(w4.T, dZ4)
    # delta3 = np.multiply(bruh, delta3)

    dw4 = (1. / m) * np.matmul(dZ4, A3.T)
    # db4 = (1./m) * np.sum(dZ4, axis=1, keepdims=True)

    # dA3 = np.matmul(w4.T, dZ4)
    # dZ3 = np.multiply(dA3, reluDerivative(Z3))
    # dw3 = (2 / m) * np.matmul(
    #         dZ3, 
    #         A2.T
    #     )
    # db3 = (2 / m) * np.sum(dZ3, axis=1, keepdims=True)

    # dA2 = np.matmul(w3.T, dZ3)
    # dZ2 = np.multiply(dA2, reluDerivative(Z2))
    # dw2 = (2 / m) * np.matmul(
    #         dZ2, 
    #         A1.T
    #     )
    # db2 = (2 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # dA1 = np.matmul(w2.T, dZ2)
    # dZ1 = np.multiply(dA1, reluDerivative(Z1))
    # dw1 = (2 / m) * np.matmul(
    #         dZ1, 
    #         x_train.T
    #     )
    # db1 = (2 / m) * np.sum(dZ1, axis=1, keepdims=True)

    w4 = w4 - (learning_rate * dw4)
    # b4 = b4 - learning_rate * db4

    # w3 = w3 - learning_rate * dw3
    # b3 = b3 - learning_rate * db3
    
    # w2 = w2 - learning_rate * dw2
    # b2 = b2 - learning_rate * db2
    
    # w1 = w1 - learning_rate * dw1
    # b1 = b1 - learning_rate * db1

    print(cost)

# %%
