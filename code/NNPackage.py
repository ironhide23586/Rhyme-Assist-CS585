from LayerPackage import *
import numpy as np

class NeuralNet:
    """Class encapsulating the deep neural network"""

    layer1 = None
    #layer2 = None
    softmax_layer = None

    outs = None

    def __init__(self, num_features, num_classes, l1_neurons=10):
        self.layer1 = Layer(num_features, neurons=l1_neurons)
        self.softmax_layer = SoftmaxLayer(num_classes, l1_neurons)

    def load_train_obs(self, data):
        """Loads training observations (X)."""
        self.layer1.load_Xs(data)

    def load_train_labels(self, data):
        """Loads training labels (Y). Must be one-hot encoded"""
        self.softmax_layer.load_Ys(data)

    def load_train_data(self, x, y):
        """Loads training data (x, y)"""
        self.load_train_obs(x)
        self.load_train_labels(y)

    def train(self):
        """Runs one training iteration."""
        self.layer1.estimate_Ys()
        self.softmax_layer.estimate_Ys(z=self.layer1.neuron_outs)
        self.softmax_layer.train_layer()
        self.layer1.train_layer(self.softmax_layer.weights, self.softmax_layer.diffs)

    def predict(self, data=None):
        """Predicts the probabilities for each class."""
        if data is not None:
            self.layer1.estimate_Ys(x=data)
        else:
            self.layer1.estimate_Ys()
        self.softmax_layer.estimate_Ys(z=self.layer1.neuron_outs)
        self.outs = self.softmax_layer.neuron_outs
        return self.outs