import numpy as np


def reLu(z: np.ndarray) -> np.ndarray:
    """Basic activation function (x if x>0 else 0)"""
    return np.maximum(z,0)


def reLu_deriv(z: np.ndarray) -> np.ndarray:
    """Derivate of reLu function"""
    return (z >0).astype(int)


def leaky_relu_deriv(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU function"""
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz


"""def softmax_(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A"""


"""def softmax__(z):
    "#""Output activation function for single-label classification
    (probability between 0 and 1 for each output node, sum of nodes' probabilities = 1)"#""
    z_max = np.max(z)
    exp_z = np.exp(z - z_max) # stable softmax to reduce large numbers
    return exp_z / np.sum(exp_z)""" 


def softmax(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Output activation function for multi-label classification
    (probability between 0 and 1 for each output node independently)"""
    return 1 / (1 + np.exp(-z))