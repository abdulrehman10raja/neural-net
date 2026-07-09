"""
Runs multiple training configurations and records results for comparison.
This produces the data needed for the Week 3 Experiment Report.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from pytorch_nn.model import SimpleNN
from pytorch_nn.train import train_model, MoonsDataset
from pytorch_nn.evaluate import evaluate_model
from data.dataset import generate_dataset
from numpy_nn.metrics import accuracy, precision, recall, f1_score
from numpy_nn.visualize import plot_confusion_matrix


def run_experiment(name, hidden_size, activation, learning_rate,
                    X_train, X_test, y_train, y_test, epochs=100):
    print(f"\n{'='*50}")
    print(f"Experiment: {name}")
    print(f"Hidden size={hidden_size}, Activation={activation}, LR={learning_rate}")
    print(f"{'='*50}")

    train_dataset = MoonsDataset(X_train, y_train)
    val_dataset = MoonsDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleNN(input_size=2, hidden_size=hidden_size, output_size=1, activation=activation)
    history = train_model(model, train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        probs = model(X_tensor).numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    results = {
        "name": name,
        "hidden_size": hidden_size,
        "activation": activation,
        "learning_rate": learning_rate,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "accuracy": accuracy(y_test, preds),
        "precision": precision(y_test, preds),
        "recall": recall(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset()

    configs = [
        {"name": "A - Baseline",        "hidden_size": 8,  "activation": "relu", "learning_rate": 0.1},
        {"name": "B - Bigger Network",  "hidden_size": 32, "activation": "relu", "learning_rate": 0.1},
        {"name": "C - High LR",         "hidden_size": 8,  "activation": "relu", "learning_rate": 1.0},
    ]

    all_results = []
    for cfg in configs:
        result = run_experiment(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            **cfg
        )
        all_results.append(result)

    print("\n\n===== SUMMARY TABLE =====")
    print(f"{'Name':<20}{'Hidden':<10}{'Activation':<12}{'LR':<8}{'Acc':<8}{'F1':<8}")
    for r in all_results:
        print(f"{r['name']:<20}{r['hidden_size']:<10}{r['activation']:<12}{r['learning_rate']:<8}{r['accuracy']:<8.4f}{r['f1']:<8.4f}")

    # Side-by-side comparison of decision boundaries across all 3 configs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cfg, result in zip(axes, configs, all_results):
        model = SimpleNN(input_size=2, hidden_size=cfg["hidden_size"],
                          output_size=1, activation=cfg["activation"])
        # retrain quickly to get a model for plotting (or store models from earlier if you prefer)
        train_dataset = MoonsDataset(X_train, y_train)
        val_dataset = MoonsDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        train_model(model, train_loader, val_loader, epochs=100, learning_rate=cfg["learning_rate"])

        model.eval()
        with torch.no_grad():
            xx, yy = np.meshgrid(np.linspace(X_train[:,0].min()-0.5, X_train[:,0].max()+0.5, 200),
                                   np.linspace(X_train[:,1].min()-0.5, X_train[:,1].max()+0.5, 200))
            grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
            Z = model(grid).numpy().reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="coolwarm", edgecolors="k", s=20)
        ax.set_title(f"{cfg['name']}\n{cfg['hidden_size']} units, {cfg['activation']}, lr={cfg['learning_rate']}")

    plt.tight_layout()
    plt.savefig("reports/experiment_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved reports/experiment_comparison.png")
    plt.close()
