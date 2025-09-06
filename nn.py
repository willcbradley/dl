import numpy as np


# Initialise parameters
def init_params(layer_sizes):
    params = {}
    for l in range(1, len(layer_sizes)):
        params[f"W{l}"] = np.random.randn(layer_sizes(l), layer_sizes(l-1))
        params[f"b{l}"] = np.zeros(layer_sizes(l), 1)
    return params

# Remaining code to be written

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])