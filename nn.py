import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialise parameters
def init_params(layer_sizes):
    params = {}
    for l in range(1, len(layer_sizes)):
        params[f"W{l}"] = np.random.randn(layer_sizes[l], layer_sizes[l-1])
        params[f"b{l}"] = np.zeros(layer_sizes(l), 1)
    return params

# Forward pass
def forward_pass(X, params):
    cache = {}
    A = X.T
    cache["A0"] = A
    for l in range(1, len(params) // 2 + 1):
        W, b = params[f"W{l}"], params[f"b{l}"]
        Z  = W @ A + b
        A = sigmoid(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
        return A, cache

# Remaining code to be written

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print(np.shape(X))
print(np.shape(X.T))