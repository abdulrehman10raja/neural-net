"""
Evaluation utilities for the trained PyTorch model.
"""

import torch
import numpy as np
from numpy_nn.metrics import accuracy, precision, recall, f1_score, classification_report


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained PyTorch model on test data.

    Parameters
    ----------
    model : nn.Module (already trained)
    X_test : np.ndarray
    y_test : np.ndarray
    threshold : float

    Returns
    -------
    np.ndarray of predicted class labels (0 or 1)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = model(X_tensor).numpy().flatten()
        preds = (probs >= threshold).astype(int)

    print("=== PyTorch Model Evaluation ===")
    classification_report(y_test, preds)  # reusing your NumPy metrics module!

    return preds
