"""
Training script for the PyTorch neural network.
Demonstrates the standard PyTorch training loop pattern:
DataLoader -> forward -> loss -> backward -> optimizer step.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_nn.model import SimpleNN
from data.dataset import generate_dataset


class MoonsDataset(Dataset):
    """
    Wraps our NumPy arrays into a PyTorch Dataset so DataLoader
    can automatically batch and shuffle them during training.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (n, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.1):
    """
    Standard PyTorch training loop.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    epochs : int
    learning_rate : float

    Returns
    -------
    dict with 'train_loss' and 'val_loss' lists, for plotting later.
    """
    criterion = nn.BCELoss()  # binary cross-entropy, built into PyTorch
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()  # tells PyTorch layers like dropout/batchnorm to behave in "training mode"
        epoch_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()          # clear gradients from previous step
            y_pred = model(X_batch)         # forward pass
            loss = criterion(y_pred, y_batch)
            loss.backward()                 # backward pass - autograd computes all gradients
            optimizer.step()                # update weights using those gradients

            epoch_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # --- Validation phase ---
        model.eval()  # switches to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad():  # don't track gradients - saves memory, we're not training here
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    return history


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset()

    train_dataset = MoonsDataset(X_train, y_train)
    val_dataset = MoonsDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleNN(input_size=2, hidden_size=8, output_size=1)
    history = train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.1)

    torch.save(model.state_dict(), "pytorch_nn/trained_model.pth")
    print("Model saved to pytorch_nn/trained_model.pth")
