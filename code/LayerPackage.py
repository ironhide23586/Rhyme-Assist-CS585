import numpy as np
import copy
from sklearn.preprocessing import PolynomialFeatures

class Layer(object):
    """Class encapsulating the functionality of a layer pertaining to a neural network."""

    """Number of Neurons in the layer."""
    num_neurons = 0

    """Number of inputs accepted by each neuron in the layer from the previous layer, including the bias."""
    num_inputs_per_neuron = 0
    
    """Learning Rate for training."""
    learn_rate = 0

    """Momentum for training to be used for dampening of the training process."""
    momentum = 0

    """Outputs produced by each neuron in the layer for each observation."""
    neuron_outs = None

    """2D weight matrix where the ith row contains the weights of the inputs to the ith neuron in the layer including the bias weight."""
    weights = None

    """Differences between the actual outputs and predicted outputs."""
    diffs = None

    """Contains the change in weights caused by previous training iteration. Used to incorporate momentum in training."""
    __prev_delta_weights = None
    
    """2D matrix containing inputs to the layer from previous layer. Each row is an observation and each column is a feature."""
    x = None
    
    """Desired outputs of each neuron. It is utilized to train the weights."""
    y = None

    __poly = None

    def __init__(self, max_num_inputs, neurons=10, learnRate=.001, momentum=.005, randomStartWeights=False):
        """Constructor for the Layer class.
        Args:
            max_num_inputs - Maximum number of inputs to each neuron in the layer from previous layer. 
                             (Bias not to be counted)
            neurons - Number of neurons in the layer.
            learnRate - Learning rate to be used for training.
            momentum - Momentum to be used for training.
            act_func - Activation function pertaining to each neuron.
            randomStartWeights - Intiial weights randomly set if this is true, else all are set to zero by default.
        """
        self.num_neurons = neurons
        self.learn_rate = learnRate
        self.momentum = momentum
        self.num_inputs_per_neuron = max_num_inputs + 1 #1 is added to incorporate the bias weight.

        if randomStartWeights is True:
            self.weights = np.random.rand(self.num_neurons, self.num_inputs_per_neuron)
        else:
            self.weights = np.zeros([self.num_neurons, self.num_inputs_per_neuron])

        self.__prev_delta_weights = np.zeros([self.num_neurons, self.num_inputs_per_neuron])
        self.__poly = PolynomialFeatures(1)


    def load_Xs(self, x):
        """Loads outputs from the previous layer onto this layer.
        Args:
            x - A 2D matrix where each row is an separate input from the previous layer & each column is a feature.
        """
        self.x = copy.deepcopy(self.__poly.fit_transform(x))   


    def load_Ys(self, y):
        """Loads outputs to be fed to the next layer, to be used for training through backpropagation of current layer.
        Args:
            y - A 2D array containing the outputs to be fed to the next layer. Each row corresponds to the neuron outputs
                of each observation.
        """
        self.y = copy.deepcopy(y)


    def estimate_Ys(self, x=None):
        """Uses the current weights to estimate the outputs of the layer pertaining to the given input from the previous layer.
        Args:
            x - A 2D matrix where each row is an separate input from the previous layer & each column is a feature.
                If None (default), the values previously loaded onto the layer will be used to produce the outputs.
        """
        if x is not None:
            self.load_Xs(x)
        self.neuron_outs = np.dot(self.x, self.weights.T)
        self.neuron_outs[self.neuron_outs < -20] = -20 #Checking exponential powers to avoid overflow while computing sigmoid.
        self.neuron_outs[self.neuron_outs > 7] = 7
        self.neuron_outs = 1 / (1 + np.exp(-self.neuron_outs))


    def train_layer(self, v, diffs):
        """Trains the layer only if this layer is right beneath the output Softmax Layer.
        Args:
            v - Weight matrix of the neuron layer in front.
            diffs - Difference between the estimated values from the output layer and the true values (One Hot Encoding)
        """
        diffs_w_0 = np.dot(diffs, v)
        z_coeffs = self.neuron_outs * (1 - self.neuron_outs)
        diffs_w_1 = np.array([[self.x[i] * z_coeffs[i][j] for i in xrange(self.x.shape[0])] for j in xrange(self.num_neurons)])
        delta_w = np.array([np.sum([diffs_w_0[i, j] * diffs_w_1[j][i] for i in xrange(self.x.shape[0])], axis=0) for j in xrange(self.num_neurons)])
        self.weights -= delta_w * self.learn_rate + self.__prev_delta_weights * self.momentum
        self.__prev_delta_weights = copy.deepcopy(delta_w)


    def train_deep_layer(self, v, diffs, z, w):
        #z = self.__poly.fit_transform(z)
        z_coeffs = z * (1 - z)
        #diffs_w_0 = np.dot(z_coeffs, w)
        #diffs_w_1 = np.dot(diffs_w_0, v.T)

        diffs_w_0 = np.array([[[np.sum([w[:, j] * z_coeffs[i] * v[l][1:]]) for j in xrange(1, w.shape[1])] for l in xrange(v.shape[0])] for i in xrange(self.x.shape[0])])
        diffs_w_1 = np.sum([np.sum([diffs[i][j] * diffs_w_0[i][j] for j in xrange(diffs.shape[1])], axis=0) for i in xrange(self.x.shape[0])], axis=0)

        q_coeffs = self.neuron_outs * (1 - self.neuron_outs)
        #diffs_w_2 = np.array([np.array([q_coeffs[i][j] * self.x[i] for i in xrange(self.x.shape[0])]) for j in xrange(q_coeffs.shape[1])])
        diffs_w_2 = np.array([np.sum([q_coeffs[i][j] * self.x[i] for i in xrange(self.x.shape[0])], axis=0) for j in xrange(q_coeffs.shape[1])])
        
        delta_w = np.array([diffs_w_1[i] * diffs_w_2[i] for i in xrange(diffs_w_1.shape[0])])
        self.weights -= delta_w * self.learn_rate + self.__prev_delta_weights * self.momentum
        self.__prev_delta_weights = copy.deepcopy(delta_w)



class SoftmaxLayer:
    """Class encapsulating the functionality of the final softmax layer pertaining to a neural network."""

    num_neurons = None
    neuron_outs = None
    learn_rate = 0.
    momentum = 0.
    num_inputs_per_neuron = 0
    z = None
    y = None
    diffs = None
    weights = None
    __prev_delta_weights = None
    __poly = None

    def __init__(self, num_outs, max_num_inputs, learnRate=.0001, momentum=.0005, randomStartWeights=False):
        self.neuron_outs = np.zeros(num_outs)
        self.num_neurons = num_outs
        self.learn_rate = learnRate
        self.momentum = momentum
        self.num_inputs_per_neuron = max_num_inputs + 1 #1 is added to incorporate the bias weight.

        if randomStartWeights is True:
            self.weights = np.random.rand(self.num_neurons, self.num_inputs_per_neuron)
        else:
            self.weights = np.ones([self.num_neurons, self.num_inputs_per_neuron])

        self.__prev_delta_weights = np.zeros([self.num_neurons, self.num_inputs_per_neuron])
        self.__poly = PolynomialFeatures(1)


    def load_Zs(self, z):
        self.z = copy.deepcopy(self.__poly.fit_transform(z))


    def load_Ys(self, y):
        self.y = copy.deepcopy(y)


    def softmax(self, arr, j):
        if arr[j] < 10:
            res = np.exp(arr[j]) / np.sum(np.exp(arr))
        else:
            arr -= arr.max()
            res = np.exp(arr[j]) / np.sum(np.exp(arr))
        return res


    def estimate_Ys(self, z=None):
        if z is not None:
            self.load_Zs(z)
        prods = np.dot(self.z, self.weights.T)
        self.neuron_outs = np.array([[self.softmax(prod, i) for i in xrange(prods.shape[1])] for prod in prods])


    def train_layer(self):
        self.diffs = self.neuron_outs - self.y
        delta_weights = np.array([np.sum([d[i] * self.z[i] for i in xrange(d.shape[0])], axis=0) for d in self.diffs.T])
        self.weights -= delta_weights * self.learn_rate + self.__prev_delta_weights * self.momentum
        self.__prev_delta_weights = copy.deepcopy(delta_weights)