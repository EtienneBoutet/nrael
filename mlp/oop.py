import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

np.random.seed(420)

class Layer():
    def __init__(self):
        pass
    
    def get_weights(self):
        pass

    def set_weights(self):
        pass

    def forward_propagation(self, inputs):
        # abstract
        pass

class Flatten():
    def __init__(self, data_format=None):
        self.data_format = data_format

    def forward_propagation(self, inputs):
        shape = self.data_format
        inputs = inputs.reshape((inputs.shape[0], shape[0] * shape[1])).T
        return inputs

class Dense():
        def __init__(self, units, activation):
            self.units = units
            self.activation = activation
            self.weights = weights
            self.biases = biases

        def forward_propagation(self, inputs):
            if not self.weights:
                self.weights = np.random.randn(self.units, inputs.shape[0])
            if not self.biases:
                self.biases = np.zeros((self.units), 1)

            z = np.matmul(self.weights, inputs) + self.biases
            
            if (self.activation) == 'sigmoid':
                a = self.sigmoid(z)
            elif (self.activation) == 'softmax':
                a = self.softmax(z)

            return a
            

        def __sigmoid(self, z):
            a = 1. / (1 + np.exp(-z))
            return a

        def __softmax(self, z):
            a = np.exp(z) / np.sum(np.exp(z), axis=0)
            return a

class Model:
    def __init__(self):
        self.layers = []
        self.learning_rate = 1

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self):
        pass

    def train(self, train_images, train_labels, epochs):
        layer_input = train_images
    
        # Every layer handle his own forward and back propagation.

        # F-prop
        for layer in self.layers:
            layer_input = layer.forward_propagation(layer_input)
        
        # B-prop


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

model = Model()
model.add_layer(Flatten((28, 28)))

x_train, y_train, x_test, y_test = load_dataset()
x_train, x_test = prep_pixels(x_train, x_test)

model.train(x_train, y_train, 1)

#model.add_layer(Dense(64, activation='sigmoid'))
#model.add_layer(Dense(10, activation='softmax'))
