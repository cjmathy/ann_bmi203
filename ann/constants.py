
N_FEATURES = 8
N_HIDDEN = 3
N_OUTPUT = 8
N_SAMPLES_PER_INPUT = 8

MAX_ITERATIONS = 10
ALPHA = 1
LAMBDA = 0.001
EPSILON = 0.0001

# X is a (N_FEATURES, N_SAMPLES) matrix (278 samples, pos and neg), 34 features
# W0          (N_HIDDEN, N_FEATURES)
# dot(W0, X)  (N_HIDDEN, N_SAMPLES)
# b0          (N_HIDDEN, N_SAMPLES)
# z1, a1      (N_HIDDEN, N_SAMPLES)
# W1          (N_OUTPUT, N_HIDDEN)
# dot(W1, a1) (N_OUTPUT, N_SAMPLES)
# b1          (N_OUTPUT, N_SAMPLES)
# z2, h       (N_OUTPUT, N_SAMPLES)
# y           (N_OUTPUT, N_SAMPLES)
