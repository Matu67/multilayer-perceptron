import numpy as np

def sigmoid(X):
    return (1 / (1 + np.exp(-X)))

def reLU(X):
    return np.maximum(0, X)

def activation_func(z):
    return sigmoid(z)
