from ann import constants as c
import numpy as np
import scipy.special


class NeuralNet(object):
    """
    self.N_NODES = list, number of nodes in each layer
    self.N_LAYERS = int, number of layers in network
    self.weights = list of numpy arrays, Wij is the weight from node j in layer l to node i in layer l+1
    self.biases = list of numpy arrays, bij is ith element has biases from layer i->i+1
    self.activations = list of numpy arrays, containing activation values for each node. used for backpropagation and prediction (activations[-1] will be the hypothesis of the network)
    """

    def __init__(self, n_nodes, n_samples_per_input=1):
        self.N_NODES = n_nodes
        self.N_LAYERS = len(n_nodes)
        self.N_SAMPLES_PER_INPUT = n_samples_per_input
        self.weights = []
        self.biases = []
        self.activations = []

        # initialize the weights to small random values, initialize the biases to 0
        for i in xrange(self.N_LAYERS-1):
            w = np.random.normal(scale=0.01,
                                 size=(self.N_NODES[i+1], self.N_NODES[i]))
            self.weights.append(w)
            b = np.zeros((self.N_NODES[i+1], self.N_SAMPLES_PER_INPUT))
            self.biases.append(b)

    def train(self, samples, labels, method="batch"):
        """train a neuralnet as an autoencoder (with a design matrix input), with batch gradient descent, or with stochastic gradient descent"""

        if method is "autoencoder":
            error = []
            for iteration in xrange(c.MAX_ITERATIONS):
                predictions = self.test(samples, method="design")
                e = np.linalg.norm(predictions-samples)
                error.append(e)

                # test for convergence
                if (e - error[iteration-1] < c.EPSILON) and e < 0.05:
                    return error, iteration+1

                deltas = self.reset_deltas()
                self.feedforward(samples)
                gradients = self.backpropagate(labels)
                self.update_deltas(deltas, gradients)
                self.update_parameters(deltas, samples, stochastic=True)
            return error, c.MAX_ITERATIONS

        if method is "batch":
            for iteration in xrange(c.MAX_ITERATIONS):
                deltas = self.reset_deltas()
                for i, sample in enumerate(samples):
                    sample = sample.reshape((c.N_FEATURES, 1))
                    self.feedforward(sample)
                    gradients = self.backpropagate(labels[i])
                    self.update_deltas(deltas, gradients)
                self.update_parameters(deltas, samples)

        if method is "stochastic":
            for iteration in xrange(c.MAX_ITERATIONS):
                deltas = self.reset_deltas()
                i = np.random.randint(0, samples.shape[0])
                self.feedforward(samples[i])
                gradients = self.backpropagate(labels[i])
                self.update_parameters(deltas, samples, stochastic=True)

    def feedforward(self, a):
        """takes an input numpy array (a) and feeds it through the neural network. Clips values to avoid precision errors."""

        # first activation is the input signal
        self.activations = [a]

        # compute activations of each layer
        for l in xrange(self.N_LAYERS-1):
            w = self.weights[l]
            b = self.biases[l]
            z = np.dot(w, a) + b
            a = scipy.special.expit(z)
            self.activations.append(a)

        # clip values for precision
        for array in self.activations:
            array[np.abs(array) < c.EPSILON] = 0.
            array[np.abs(array) > 1-c.EPSILON] = 1.

    def backpropagate(self, y):
        """takes in a known set of labels, and backpropagates the error through the network. Assumes a sigmoid activator function, a mean squared error loss function, and a ridge regression term (lambda is multiplied by the l2-norm of w)"""
        h = self.activations[-1]
        error = y - h
        fprime = np.multiply(h, 1-h)
        delta = - np.multiply(error, fprime)
        deltas = [delta]

        for l in xrange(self.N_LAYERS-2, 0, -1):
            a = self.activations[l]
            wd = np.dot(self.weights[l].T, delta)
            fprime = np.multiply(a, 1-a)
            delta = np.multiply(wd, fprime)
            deltas.insert(0, delta)

        gradW, gradB = [], []

        for l in xrange(self.N_LAYERS-1):
            d = deltas[l]
            a = self.activations[l]
            partial = np.dot(d, a.T)
            gradW.append(partial)
            gradB.append(d)

        return gradW, gradB

    def reset_deltas(self):
        """resets the deltas, which are used to adjust the network parameters w and b"""
        deltaW, deltaB = [], []
        for w in self.weights:
            deltaW.append(np.zeros(w.shape))
        for b in self.biases:
            deltaB.append(np.zeros(b.shape))
        return deltaW, deltaB

    def update_deltas(self, deltas, gradients):
        """updates the deltas according to the gradients computed through backpropagation"""
        deltaW, deltaB = deltas
        gradW, gradB = gradients
        for l in xrange(self.N_LAYERS-1):
            deltaW[l] = deltaW[l] + gradW[l]
            deltaB[l] = deltaB[l] + gradB[l]

    def update_parameters(self, deltas, samples, stochastic=False):
        """updates system parameters using computed delta values. Supports batch and stochastic gradient descent"""
        deltaW, deltaB = deltas
        alpha = c.ALPHA
        lam = c.LAMBDA
        for l in xrange(self.N_LAYERS-1):
            w, b = self.weights[l], self.biases[l]
            dW, dB = deltaW[l], deltaB[l]

            if stochastic is True:
                self.weights[l] = w - alpha*(dW + lam*w)
                self.biases[l] = b - alpha*dB
            else:
                m = samples.shape[0]
                self.weights[l] = w - alpha*(dW/m + lam*w)
                self.biases[l] = b - alpha*(dB/m)

    def test(self, X, method="sample"):
        """Feeds a test numpy array through a network, and returns the output hypothesis. Test array can be fed through at once as a design matrix (useful for the autoencoder) or can be fed through sample by sample."""
        if method is "design":
            self.feedforward(X)
            return self.activations[-1]

        if method is "sample":
            predictions = np.zeros((X.shape[0],))
            for i, sample in enumerate(X):
                sample = sample.reshape((c.N_FEATURES, 1))
                self.feedforward(sample)
                predictions[i] = self.activations[-1]
            return predictions
