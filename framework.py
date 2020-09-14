# Sources:
# Understanding Backpropogation Algorithm - https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd
# 3Blue1Brown Neural Network Video Series - https://www.3blue1brown.com/neural-networks

import numpy as np
from foldr import foldr
from activation_funcs import activation_func
import pandas as pd

class Layer():
    def __init__(self, nodes, biases, weights):
        self.a = nodes
        self.b = biases
        self.W = weights
        self.z = np.zeros(nodes.shape)
        self.n = len(nodes)

class Network():
    def print_nodes(self):
        for layer in self.layers:
            print(pd.DataFrame(layer.a))
    
    def print_biases(self):
        for layer in self.layers:
            print(pd.DataFrame(layer.b))

    def print_weights(self):
        for layer in self.layers:
            print(pd.DataFrame(layer.W))
    
    def forward_prop(self):
        for i in range(1, len(self.layers)):
            self.layers[i].z = np.add(np.matmul(self.layers[i - 1].W, self.layers[i - 1].a), self.layers[i - 1].b)
            self.layers[i].a = activation_func(self.layers[i].z)
    
    def calc_der_C_wrt_a(self, y_expected):
        der_C_wrt_a = [2 * (self.layers[-1].a - y_expected)]
        for i in reversed(range(0, len(self.layers) - 1)):
            temp = np.array([])
            for k in range(self.layers[i].n):
                weights = self.layers[i].W[:, k]
                deriv_z = activation_func(self.layers[i + 1].z, deriv=True)
                deriv_c = der_C_wrt_a[0]
                temp = np.insert(temp, 0, np.sum(weights * deriv_z * deriv_c))
            der_C_wrt_a.insert(0, temp)
        return der_C_wrt_a

    def back_prop(self, y_expected):
        der_C_wrt_a = self.calc_der_C_wrt_a(y_expected)
        gradient_C = []
        for i in range(0, len(self.layers) - 1):
            temp = np.matmul(np.resize(self.layers[i + 1].z, (self.layers[i + 1].n, 1)), np.resize(self.layers[i].a, (1, self.layers[i].n)))
            gradient_C.append(temp * der_C_wrt_a[i])
            gradient_C.append(temp)
        return gradient_C
    
    def input(self, x):
        self.layers[0].a = x
        self.layers[0].z = np.zeros(self.layers[0].n)

    def __init__(self, x, hiddens_struct, y):
        output = [Layer(np.zeros(y), np.array([]), np.array([]))]
        self.layers = [Layer(x, np.random.rand(hiddens_struct[0]) * 20 - 10, np.random.rand(hiddens_struct[0], len(x)))] + \
                      foldr(lambda a, b: [Layer(np.zeros(a), np.random.rand(b[0].n) * 20 - 10, np.random.rand(b[0].n, a))] + b, output, hiddens_struct)
