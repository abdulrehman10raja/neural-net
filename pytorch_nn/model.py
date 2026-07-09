"""
Neural network model defined using PyTorch's nn.Module.
Same architecture as the NumPy version: Input -> Hidden -> Output (Sigmoid)
Activation function is now configurable for experimentation.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
