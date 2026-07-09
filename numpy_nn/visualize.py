"""
Visualization utilities - plots the decision boundary a trained
network has learned, overlaid on the actual data points.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None):
    """
    Plot the decision boundary of a trained model over a 2D dataset.

    Parameters
    ----------
    model : object with a .predict(X) method
        Trained NeuralNetwork instance.
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples,)
    title : str
    save_path : str or None
        If given, saves the plot to this file instead of just showing it.
    """
    # Create a grid of points covering the data range, with a small margin
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Predict the class for every point on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the filled contour (the "regions" the model has learned)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    # Plot the actual data points on top
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_loss_curve(loss_history, title="Training Loss", save_path=None):
    """Plot loss over epochs to visualize training progress."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()
