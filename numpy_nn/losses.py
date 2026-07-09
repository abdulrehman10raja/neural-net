"""
Loss functions for binary classification, implemented from scratch.
Binary Cross-Entropy (BCE) measures how well predicted probabilities
match the true 0/1 labels - it punishes confident wrong predictions
much more heavily than uncertain ones.
"""

import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss, averaged across all samples.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels, each 0 or 1.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted probabilities (output of sigmoid), each between 0 and 1.

    Returns
    -------
    float
        Average loss across all samples. Lower is better.
    """
    # Clip predictions to avoid log(0), which is undefined (-infinity)
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def binary_cross_entropy_derivative(y_true, y_pred):
    """
    Derivative of BCE loss with respect to y_pred.
    This tells backpropagation which direction to push the predictions
    to reduce the loss.

    Formula: dL/dy_pred = -(y_true/y_pred) + (1-y_true)/(1-y_pred)
    Simplifies nicely when combined with sigmoid derivative during backprop.
    """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
