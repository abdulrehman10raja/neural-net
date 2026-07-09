"""
Feedforward Neural Network implemented from scratch using only NumPy.
Architecture: Input -> Hidden Layer (ReLU) -> Output Layer (Sigmoid)
Trained using manual backpropagation and gradient descent.
"""

import numpy as np
from numpy_nn.activations import relu, relu_derivative, sigmoid
from numpy_nn.losses import binary_cross_entropy


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1, random_state=42):
        """
        Initialize weights and biases for a 2-layer neural network.

        Parameters
        ----------
        input_size : int
            Number of input features (2 for our moons dataset).
        hidden_size : int
            Number of neurons in the hidden layer (adjustable).
        output_size : int
            Number of output neurons (1 for binary classification).
        random_state : int
            Seed for reproducible weight initialization.
        """
        rng = np.random.default_rng(random_state)

        # He initialization for W1 (good pairing with ReLU) -
        # scales weights based on layer size to prevent vanishing/exploding values
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # Xavier-style initialization for W2 (good pairing with Sigmoid)
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Cache to store intermediate values needed during backpropagation
        self.cache = {}

    def forward(self, X):
        """
        Forward pass: compute predictions given input X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)

        Returns
        -------
        np.ndarray, shape (n_samples, output_size)
            Predicted probabilities.
        """
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)

        # Save these - backward() needs them to compute gradients
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2

    def backward(self, y_true, learning_rate=0.1):
        """
        Backward pass: compute gradients via backpropagation and
        update weights using gradient descent.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples, 1)
            True labels.
        learning_rate : float
            Step size for gradient descent updates.
        """
        X = self.cache["X"]
        A1 = self.cache["A1"]
        Z1 = self.cache["Z1"]
        A2 = self.cache["A2"]
        m = X.shape[0]  # number of samples, used to average gradients

        # --- Output layer gradients ---
        # This is the simplified derivative of BCE loss combined with sigmoid:
        # dL/dZ2 = A2 - y_true  (a well-known neat result)
        dZ2 = A2 - y_true
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # --- Hidden layer gradients (push error backward through W2) ---
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # --- Update weights: move opposite to the gradient direction ---
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """
        Full training loop: forward pass -> loss -> backward pass, repeated.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
        epochs : int
            Number of times to loop over the full training set.
        learning_rate : float
        verbose : bool
            If True, prints loss every 100 epochs.

        Returns
        -------
        list of float
            Loss recorded at each epoch (useful for plotting later).
        """
        y = y.reshape(-1, 1)  # ensure shape (n_samples, 1) for matrix math
        loss_history = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred)
            loss_history.append(loss)

            self.backward(y, learning_rate=learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

        return loss_history

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1) for input X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)
        threshold : float
            Probability cutoff - above this = class 1, below = class 0.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probs = self.forward(X)
        return (probs >= threshold).astype(int).flatten()
