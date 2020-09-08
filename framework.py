import numpy as np
from foldr import foldr
from activation_funcs import activation_func

class Layer():
    def __init__(self, nodes, biases, weights):
        self.a = nodes
        self.b = biases
        self.W = weights
        self.n = len(nodes)

class Network():
    def print_nodes(self):
        for layer in self.layers:
            print(layer.a)
    
    def print_biases(self):
        for layer in self.layers:
            print(layer.b)

    def print_weights(self):
        for layer in self.layers:
            print(layer.W)
    
    def fill_nodes(self):
        for i in range(1, len(self.layers)):
            self.layers[i].a = activation_func(np.add(np.matmul(self.layers[i - 1].W, self.layers[i - 1].a), self.layers[i - 1].b))
    
    def __init__(self, x, hiddens_struct, y):
        output = [Layer(np.zeros(y), np.array([]), np.array([]))]
        self.layers = [Layer(x, np.random.rand(hiddens_struct[0]) * 20 - 10, np.random.rand(hiddens_struct[0], len(x)))] + \
                      foldr(lambda a, b: [Layer(np.zeros(a), np.random.rand(b[0].n) * 20 - 10, np.random.rand(b[0].n, a))] + b, output, hiddens_struct)

