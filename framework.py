import numpy as np
from foldr import foldr

HIDDENS_STRUCT = [16, 16]

class Layer():
    def __init__(self, nodes, biases, weights):
        self.nodes = np.array(nodes)
        self.biases = np.array(biases)
        self.weights = np.array(weights)
        self.n = len(nodes)
        self.b = len(biases)
        self.w = len(weights)

class Network():
    
    # We assume that structure has at least 3 elements
    def __init__(self, x, hiddens_struct, y):
        self.x = x
        self.input = Layer(x, np.random.rand(hiddens_struct[0]), 
                              np.random.rand((len(x), hiddens_struct[0])))
        self.y = y
        self.output = Layer(np.zeros(y.shape), [], [])
        self.hiddens = foldr(lambda a, b: b.insert(0, 
                                                  Layer(np.random.rand(a), 
                                                        np.random.rand(b[0].n), 
                                                        np.random.rand((a.n, b[0].n)))), 
                            [self.output], hiddens_struct)
        
    


