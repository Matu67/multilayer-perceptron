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
    def calc_a(self, l, prev_l):
        self.a = np.matmul(

    def __init__(self, nodes, biases, weights):
        self.a = np.array(nodes)
        self.b = np.array(biases)
        self.w = np.array(weights)
        self.n = len(nodes)

class Network():
    def fill_nodes(self):
        for i, l in enumerate(self.hiddens):
            if i is not 0:
                

    # We assume that structure has at least 3 elements
    def __init__(self, x, hiddens_struct, y):
        self.x = x
        self.input = Layer(x, np.random.rand(hiddens_struct[0]), 
                              np.random.rand((len(x), hiddens_struct[0])))
        self.y = y
        self.output = Layer(np.zeros(y.shape), [], [])
        self.hiddens = foldr(lambda a, b: b.insert(0, 
                                                  Layer(np.zeros(y.shape), 
                                                        np.random.rand(b[0].n), 
                                                        np.random.rand((a.n, b[0].n)))), 
                            [self.output], hiddens_struct)
        
    


