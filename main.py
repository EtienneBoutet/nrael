from matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
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

x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = prep_pixels(x_train, x_test)

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels)).T
x_test = x_test.reshape((x_test.shape[0], num_pixels)).T

# %%
# Fonction d'activation
def sigmoid(z):
    s = 1. / (1 + np.exp(-z))
    return s


# Fonction de loss avec moyenne de chacun
def multiclass_cross_entropy(y, y_hat):
    return -(1 / 60000) * np.sum(np.multiply(y, np.log(y_hat)))


# %%
# Première hidden layer de 64 neuronnes

nnodes_h1 = 64
w1 = np.random.randn(nnodes_h1, x_train.shape[0])
b1 = np.zeros((nnodes_h1, 1))

# Couche de ouput de 10 neuronnes
nnode_o1 = 10
w2 = np.random.randn(nnode_o1, nnodes_h1)
b2 = np.zeros((nnode_o1, 1))

# Nombre de données d'entraînement
m = 60000
learning_rate = 1
# %%

costs = []
accuracy = []
epochs = 1000

# ACTUAL TRAINING
for i in range(epochs):
    Z1 = np.matmul(w1, x_train) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(w2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    cost = multiclass_cross_entropy(y_train.T, A2)
    costs.append(cost)

    # Back-propagation
    dZ2 = A2 - y_train.T
    dW2 = (1. / m) * np.matmul(dZ2, A1.T)
    db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(w2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1. / m) * np.matmul(dZ1, x_train.T)
    db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

    w2 = w2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    w1 = w1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    accuracy.append(accuracy_score(np.argmax(A2, axis=0), np.argmax(y_train.T, axis=0)))

    if i % 100 == 0:
        print("Epoch", i, "cost: ", cost)

# %%
# Tester avec données de test
Z1 = np.matmul(w1, x_test) + b1
A1 = sigmoid(Z1)
Z2 = np.matmul(w2, A1) + b2
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

# Pour métriques
predictions = np.argmax(A2, axis=0)
labels = np.argmax(y_test.T, axis=0)


# %%
# MÉTRIQUES

def confusion_matrix(predictions, labels):
    k = len(np.unique(labels))
    result = np.zeros((k, k))
    for i in range(len(labels)):
        result[labels[i]][predictions[i]] += 1

    return result


def plot_cost_function(costs):
    k = len(costs)
    _, ax = plt.subplots()
    ax.plot(list(range(1, k + 1)), costs)


def plot_accuracy(accuracy):
    k = len(accuracy)
    _, ax = plt.subplots()
    ax.plot(list(range(1, k + 1)), accuracy)


# %%
print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
# %%
plot_cost_function(costs)
# %%
plot_accuracy(accuracy)
# %%
# %%
# %%
# %%
# %%
# %%
# %%
