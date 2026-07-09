import numpy as np
from numpy_nn.activations import relu, relu_derivative, sigmoid, sigmoid_derivative


def test_relu_positive():
    assert np.array_equal(relu(np.array([1, 2, 3])), np.array([1, 2, 3]))


def test_relu_negative():
    assert np.array_equal(relu(np.array([-1, -2, -3])), np.array([0, 0, 0]))


def test_relu_mixed():
    assert np.array_equal(relu(np.array([-2, 0, 2])), np.array([0, 0, 2]))


def test_relu_derivative_positive():
    assert np.array_equal(relu_derivative(np.array([1, 2])), np.array([1.0, 1.0]))


def test_relu_derivative_negative():
    assert np.array_equal(relu_derivative(np.array([-1, -2])), np.array([0.0, 0.0]))


def test_sigmoid_zero():
    assert np.isclose(sigmoid(np.array([0]))[0], 0.5)


def test_sigmoid_large_positive():
    assert sigmoid(np.array([1000]))[0] > 0.99


def test_sigmoid_large_negative():
    assert sigmoid(np.array([-1000]))[0] < 0.01


def test_sigmoid_output_range():
    values = sigmoid(np.array([-10, -1, 0, 1, 10]))
    assert np.all(values >= 0) and np.all(values <= 1)


def test_sigmoid_derivative_at_zero():
    assert np.isclose(sigmoid_derivative(np.array([0]))[0], 0.25)
