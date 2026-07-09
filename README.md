# Neural Network Implementation (Week 3)

This project contains implementations of a neural network from scratch using `numpy` and a deep learning model using `pytorch`.

## Project Structure

```text
week3-neural-network/
│
├── numpy_nn/                 # Neural Network from scratch using NumPy
│   ├── __init__.py
│   ├── network.py            # Neural network architecture, layers, forward/backward passes
│   ├── activations.py        # Activation functions (ReLU, Sigmoid, Softmax, etc.) and derivatives
│   ├── losses.py             # Loss functions (MSE, Cross Entropy, etc.) and derivatives
│   ├── metrics.py            # Evaluation metrics (Accuracy, F1-Score, etc.)
│   └── visualize.py          # Plotting loss curves, decision boundaries, etc.
│
├── pytorch_nn/               # Neural Network implementation using PyTorch
│   ├── __init__.py
│   ├── model.py              # PyTorch model definitions
│   ├── train.py              # Training loop script
│   └── evaluate.py           # Evaluation script
│
├── tests/                    # Unit tests for verification
│   ├── __init__.py
│   ├── test_activations.py
│   ├── test_losses.py
│   ├── test_network.py
│   └── test_metrics.py
│
├── data/                     # Dataset utilities
│   ├── __init__.py
│   └── dataset.py            # Data loading, generation, and preprocessing
│
├── reports/                  # Reports and analysis
│   └── experiment_report.md  # Detailed writeup of experiments and comparisons
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests to ensure everything is set up correctly:
   ```bash
   pytest
   ```
