import numpy as np
from numpy_nn.losses import binary_cross_entropy, binary_cross_entropy_derivative


def test_bce_perfect_predictions():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1 - 1e-9, 1e-9, 1 - 1e-9, 1e-9])
    loss = binary_cross_entropy(y_true, y_pred)
    assert loss < 0.01


def test_bce_wrong_predictions_high_loss():
    y_true = np.array([1, 0])
    y_pred = np.array([0.01, 0.99])
    loss = binary_cross_entropy(y_true, y_pred)
    assert loss > 2.0


def test_bce_uncertain_predictions():
    y_true = np.array([1, 0])
    y_pred = np.array([0.5, 0.5])
    loss = binary_cross_entropy(y_true, y_pred)
    assert np.isclose(loss, -np.log(0.5), atol=0.01)


def test_bce_no_nan_at_extremes():
    y_true = np.array([1, 0])
    y_pred = np.array([1.0, 0.0])
    loss = binary_cross_entropy(y_true, y_pred)
    assert not np.isnan(loss)


def test_bce_derivative_shape():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.7, 0.3, 0.9])
    grad = binary_cross_entropy_derivative(y_true, y_pred)
    assert grad.shape == y_true.shape
