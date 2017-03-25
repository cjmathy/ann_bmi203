import numpy as np
import ann.constants as c
from ann.neuralnet import NeuralNet
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics


def test_autoencoder():

    c.N_FEATURES = 8
    c.N_HIDDEN = 3
    c.N_OUTPUT = 8
    layer_sizes = [c.N_FEATURES, c.N_HIDDEN, c.N_OUTPUT]
    samples_per_input = 8

    c.EPSILON = 0.001
    c.ALPHA = 30
    c.LAMBDA = 0.0001
    training_data = np.eye(8)

    #run for 1 iteration, show that the error decreases
    c.MAX_ITERATIONS = 1

    nn = NeuralNet(layer_sizes, samples_per_input)
    error1, _ = nn.train(training_data, training_data, method="autoencoder")
    error2, _ = nn.train(training_data, training_data, method="autoencoder")
    assert error2 < error1


def test_neuralnetwork():

    c.N_FEATURES = 5
    c.N_HIDDEN = 2
    c.N_OUTPUT = 1
    c.EPSILON = 0.0001
    c.ALPHA = 30
    c.LAMBDA = 0.001
    c.MAX_ITERATIONS = 100
    layer_sizes = [c.N_FEATURES, c.N_HIDDEN, c.N_OUTPUT]

    X = np.array([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    y = np.array([1, 0])
    nn = NeuralNet(layer_sizes)
    nn.train(X, y, method="batch")
    predictions = nn.test(X, method="sample")

    # check that the neural net can make predictions between 0 and 1
    assert ((predictions <= 1).all() and (predictions >= 0).all())

    fpr, tpr, _ = sklearn.metrics.roc_curve(y, predictions)
    auc = sklearn.metrics.auc(fpr, tpr)

    # check that after initial training, the training set can be sufficiently predicted.
    assert 1 - auc < 0.01
