import numpy as np

def sigmoid(X, deriv = False):
    sigmoid = (1 / (1 + np.exp(-X)))
    if deriv:
        return sigmoid * (1 - sigmoid)
    return sigmoid

def reLU(X, deriv = False):
    if deriv:
        return np.where(X > 0, 1, 0)
    return np.maximum(0, X)

def activation_func(Z, deriv = False):
    return sigmoid(Z, deriv)
