"""
Evaluation metrics for binary classification, implemented from scratch.
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of predictions that exactly match the true labels."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """
    Of all predicted positives (1), what fraction were actually positive?
    Answers: "When the model says class 1, can I trust it?"
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall(y_true, y_pred):
    """
    Of all actual positives (1), what fraction did the model catch?
    Answers: "Did the model miss any real class-1 cases?"
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def f1_score(y_true, y_pred):
    """
    Harmonic mean of precision and recall - a single balanced score.
    Useful when you want one number that punishes extreme imbalance
    between precision and recall.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def classification_report(y_true, y_pred):
    """Print a clean summary of all metrics at once."""
    print(f"Accuracy:  {accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall:    {recall(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
