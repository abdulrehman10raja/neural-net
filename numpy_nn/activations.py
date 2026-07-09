"""
Activation functions and their derivatives, implemented from scratch.
These introduce non-linearity into the network - without them, stacking
layers would be mathematically equivalent to a single linear layer.
"""

import numpy as np


def relu(z):
    """
    ReLU: Rectified Linear Unit.
    Rule: output = z if z > 0, else 0.
    Used in hidden layers - fast to compute, avoids vanishing gradients.
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU.
    Slope is 1 where z > 0, and 0 where z <= 0.
    Used during backpropagation to know how much to adjust weights.
    """
    return (z > 0).astype(float)


def sigmoid(z):
    """
    Sigmoid: squishes any real number into range (0, 1).
    Used in the output layer for binary classification -
    output can be interpreted as "probability of class 1".
    """
    # Clip z to avoid overflow errors in np.exp for very large negative numbers
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid.
    Formula: sigmoid(z) * (1 - sigmoid(z))
    """
    s = sigmoid(z)
    return s * (1 - s)
