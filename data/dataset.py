"""
Dataset generation module for Week 3 - Neural Networks.
Generates a synthetic binary classification dataset (two interleaving moons)
that is NOT linearly separable - this is what makes a neural network
necessary instead of simple logistic regression.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_dataset(n_samples=1000, noise=0.2, test_size=0.2, random_state=42):
    """
    Generate the moons dataset and split into train/test sets.

    Parameters
    ----------
    n_samples : int
        Total number of data points to generate.
    noise : float
        Standard deviation of Gaussian noise added to the data.
        Higher noise = harder to classify perfectly.
    test_size : float
        Fraction of data reserved for testing (0.2 = 20%).
    random_state : int
        Seed for reproducibility - same seed always gives same data.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        X arrays have shape (n_samples, 2) -> two input features (x, y coords)
        y arrays have shape (n_samples,)   -> 0 or 1 class label
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    X_train, X_test, y_train, y_test = generate_dataset()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Class balance in y_train: {np.bincount(y_train)}")
