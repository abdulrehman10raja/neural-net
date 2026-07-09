import numpy as np
from numpy_nn.metrics import accuracy, precision, recall, f1_score


def test_accuracy_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_half():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.5


def test_precision_no_false_positives():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert precision(y_true, y_pred) == 1.0


def test_precision_with_false_positives():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    assert precision(y_true, y_pred) == 0.5


def test_precision_no_predicted_positives():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert precision(y_true, y_pred) == 0.0


def test_recall_catches_all():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert recall(y_true, y_pred) == 1.0


def test_recall_misses_some():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    assert recall(y_true, y_pred) == 0.5


def test_f1_balanced():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    assert np.isclose(f1_score(y_true, y_pred), 0.6667, atol=0.001)


def test_f1_zero_when_no_positives_predicted():
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    assert f1_score(y_true, y_pred) == 0.0
