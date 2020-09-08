import numpy as np
from foldr import foldr

HIDDENS_STRUCT = [16, 16]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reLU(x):
    return np.maximum(0, x)

def activation_func(z):
    return reLU(z)

class Layer():
    def __init__(self, nodes, biases, weights):
        self.a = np.array(nodes)
        self.b = np.array(biases)
        self.W = np.array(weights)
        self.n = len(nodes)

class Network():
    def fill_nodes(self):
        for i in range(1, len(self.layers) - 1):
            self.layers[i].a = activation_func(np.add(np.matmul(self.layers[i - 1].W, self.layers[i - 1].a), self.layers[i - 1].b))
    
    # We assume that structure has at least 3 elements
    def __init__(self, x, hiddens_struct, y):
        self.layers = [Layer(x, np.random.rand(hiddens_struct[0]), np.random.rand((len(x), hiddens_struct[0])))]
        self.layers += foldr(lambda a, b: b.insert(0, Layer(np.zeros(y.shape), np.random.rand(b[0].n), np.random.rand((a.n, b[0].n)))), [self.layers], hiddens_struct)
        self.layers += [Layer(np.zeros(y.shape), [], [])]