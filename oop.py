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
        shape = self.input_shape
        inputs = inputs.reshape((inputs.shape[0], shape[0] * shape[1])).T
        return inputs

class Dense():
        def __init__(self, units, activation, weights=None, biases=None):
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
        for layer in self.layers:
            layer_input = layer.forward_propagation(layer_input)

    def validate(self):
        pass


model = Model()
model.add_layer(Flatten((28, 28)))
