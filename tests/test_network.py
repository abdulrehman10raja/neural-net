import numpy as np
from numpy_nn.network import NeuralNetwork


def test_forward_output_shape():
    nn = NeuralNetwork(input_size=2, hidden_size=4)
    X = np.random.randn(10, 2)
    output = nn.forward(X)
    assert output.shape == (10, 1)


def test_forward_output_range():
    nn = NeuralNetwork(input_size=2, hidden_size=4)
    X = np.random.randn(10, 2)
    output = nn.forward(X)
    assert np.all(output >= 0) and np.all(output <= 1)


def test_weight_shapes():
    nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)
    assert nn.W1.shape == (2, 8)
    assert nn.b1.shape == (1, 8)
    assert nn.W2.shape == (8, 1)
    assert nn.b2.shape == (1, 1)


def test_loss_decreases_during_training():
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    nn = NeuralNetwork(input_size=2, hidden_size=8)
    loss_history = nn.train(X, y, epochs=200, learning_rate=0.1, verbose=False)

    assert loss_history[-1] < loss_history[0]


def test_predict_returns_binary_labels():
    nn = NeuralNetwork(input_size=2, hidden_size=4)
    X = np.random.randn(20, 2)
    preds = nn.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_shape_matches_input_rows():
    nn = NeuralNetwork(input_size=2, hidden_size=4)
    X = np.random.randn(15, 2)
    preds = nn.predict(X)
    assert preds.shape == (15,)


def test_network_learns_simple_pattern_well():
    np.random.seed(1)
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)

    nn = NeuralNetwork(input_size=2, hidden_size=4)
    nn.train(X, y, epochs=500, learning_rate=0.1, verbose=False)
    preds = nn.predict(X)
    acc = np.mean(preds == y)

    assert acc > 0.90
