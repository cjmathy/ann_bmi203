from neuralnet import NeuralNet
import rap1
import constants as c
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics

"""Part 1: AutoEncoder (8x8 Identity Matrix)"""

# c.N_FEATURES = 8
# c.N_HIDDEN = 3
# c.N_OUTPUT = 8
# layer_sizes = [c.N_FEATURES, c.N_HIDDEN, c.N_OUTPUT]
# samples_per_input = 8

# c.MAX_ITERATIONS = 500
# c.EPSILON = 0.001
# training_data = np.eye(8)

# alpha = [5, 10, 20, 30, 40, 50]
# lmbda = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
# plt.figure(1, figsize=(30, 15))
# for i, l in enumerate(lmbda):
#     c.LAMBDA = l
#     plt.subplot(str(231+i))
#     for a in alpha:
#         c.ALPHA = a
#         nn = NeuralNet(layer_sizes, samples_per_input)
#         error, iterations = nn.train(training_data, training_data, method="autoencoder")
#         plt.plot(range(iterations), error, label='alpha = %i' % a)
#         plt.legend(loc="lower right")
#     plt.xlabel("iterations")
#     plt.ylabel("error (2-norm of h-I)")
#     plt.title("lambda = 10^(%i)" % (i-5))
# plt.suptitle("autoencoder hypertuning, 8x8 Identity Matrix")
# plt.savefig("autoencoder_tuning.png")

"""Part 2: Rap1 Binding hypertuning"""

# c.N_FEATURES = 34
# c.N_HIDDEN = 5
# c.N_OUTPUT = 1
# c.EPSILON = 0.0001
# c.ALPHA = 30
# c.LAMBDA = 0.001
# c.MAX_ITERATIONS = 1000

# pos, neg_sim, neg_rand = rap1.prepare_data()
# X, y = rap1.build_training_set(pos, neg_sim, neg_rand)
# X, y = sklearn.utils.shuffle(X, y)

# for h in xrange(1, 21):
#     "Number of Hidden Nodes: %i" % h
#     c.N_HIDDEN = h
#     # training regime for one set of parameters, using k-fold cross-val
#     k = 5
#     kf = sklearn.model_selection.KFold(n_splits=k)
#     layer_sizes = [c.N_FEATURES, c.N_HIDDEN, c.N_OUTPUT]
#     auc_per_fold = []
#     for train, val in kf.split(X):
#         nn = NeuralNet(layer_sizes)
#         nn.train(X[train], y[train], method="batch")
#         predictions = nn.test(X[val], method="sample")
#         fpr, tpr, _ = sklearn.metrics.roc_curve(y[val], predictions)
#         auc_fold = sklearn.metrics.auc(fpr, tpr)
#         print "auc_fold is %4f" % auc_fold
#         auc_per_fold.append(auc_fold)
#     auc = float(sum(auc_per_fold))/k
#     print "average auc is:"
#     print auc

"""Part 3: Rap1 Binding Iterative Improvement"""

# c.N_FEATURES = 34
# c.N_HIDDEN = 5
# c.N_OUTPUT = 1
# c.EPSILON = 0.0001
# c.ALPHA = 30
# c.LAMBDA = 0.001
# c.MAX_ITERATIONS = 2000

# pos, neg_sim, neg_rand = rap1.prepare_data()

# pos_test = pos[100:]
# pos = pos[:100]

# X_val, y_val = rap1.build_test_set(pos_test, neg_sim, neg_rand)
# layer_sizes = [c.N_FEATURES, c.N_HIDDEN, c.N_OUTPUT]
# nn = NeuralNet(layer_sizes)

# aucs = []

# for i in xrange(20):
#     print i
#     X_trn, y_trn = rap1.build_training_set_100(pos, neg_sim, neg_rand)
#     X_trn, y_trn = sklearn.utils.shuffle(X_trn, y_trn)
#     nn.train(X_trn, y_trn, method="batch")
#     predictions = nn.test(X_val, method="sample")
#     fpr, tpr, _ = sklearn.metrics.roc_curve(y_val, predictions)
#     auc = sklearn.metrics.auc(fpr, tpr)
#     aucs.append(auc)

# plt.plot(range(20), aucs)
# plt.xlabel('Number of training sets seen')
# plt.ylabel('AUROC')
# plt.title("Test Set improvement after seeing iterative training sets")
# plt.savefig("test aucs.png")

"""Part 4: Training best network"""

# c.N_FEATURES = 34
# c.N_HIDDEN = 5
# c.N_OUTPUT = 1
# c.EPSILON = 0.0001
# c.ALPHA = 30
# c.LAMBDA = 0.001
# c.MAX_ITERATIONS = 2000

# pos, neg_sim, neg_rand = rap1.prepare_data()
# X, y = rap1.build_training_set(pos, neg_sim, neg_rand)
# X, y = sklearn.utils.shuffle(X, y)

# layer_sizes = [c.N_FEATURES, c.N_HIDDEN, c.N_OUTPUT]
# nn = NeuralNet(layer_sizes)
# nn.train(X, y, method="batch")

# test_seqs = []
# file = '/Users/cjmathy/Documents/courses/bmi203/Final-Project/ann_bmi203/rap1-lieb-test.txt'
# with open(file, 'rb') as f:
#     for seq in f:
#         test_seqs.append(seq.strip())
# test_vecs = rap1.str_to_vec(test_seqs)
# X_test = np.array(test_vecs)
# predictions = nn.test(X_test, method="sample")

# with open("predictions_mathy.txt", "w") as f:
#     for i, seq in enumerate(test_seqs):
#         print seq + "\t" + str(predictions[i]) + "\t"
#         f.write(seq)
#         f.write("\t")
#         f.write(str(predictions[i]))
#         f.write("\t")
