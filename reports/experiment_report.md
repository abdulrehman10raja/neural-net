# Week 3 Experiment Report: Neural Network Configuration Comparison

## Objective
Compare three training configurations of the PyTorch neural network on the
moons binary classification dataset, varying hidden layer size and learning
rate, to understand how each hyperparameter affects training dynamics and
final performance.

## Dataset
Synthetic "two moons" dataset (1000 samples, 20% noise), split 80/20 into
train/test sets. Chosen because it is not linearly separable, making it a
good test of whether the network is actually learning a non-linear boundary.

## Configurations Tested

| Config | Hidden Size | Activation | Learning Rate | Accuracy | F1 Score |
|--------|------------|------------|----------------|----------|----------|
| A - Baseline       | 8  | ReLU | 0.1 | 0.9800 | 0.9802 |
| B - Bigger Network | 32 | ReLU | 0.1 | 0.9850 | 0.9851 |
| C - High LR        | 8  | ReLU | 1.0 | 0.9650 | 0.9645 |

## Analysis

### Config A vs Config B: Hidden Layer Size
Increasing the hidden layer from 8 to 32 neurons improved accuracy slightly
(98.0% -> 98.5%) and reached a lower loss faster during training. More
neurons give the network more "capacity" - more combinations of lines it can
bend and combine to approximate the curved moon boundary. Since the dataset
is small and simple, the improvement was modest; on a harder dataset the
gap would likely be larger. The training loss curve for Config B also
dropped faster in early epochs (0.5376 at epoch 0 vs 0.6130 for Config A),
showing that a bigger network can start finding useful patterns sooner
simply because it has more parameters to work with.

### Config A vs Config C: Learning Rate
Config C used a learning rate 10x higher (1.0 vs 0.1) and converged to a
low loss almost immediately - by epoch 20, its loss was already lower than
Config A ever reached. However, this speed came at a cost: the validation
loss became unstable, oscillating between roughly 0.04 and 0.07 across
later epochs instead of decreasing smoothly. This is a classic sign of a
learning rate that is too large - the optimizer takes steps so big that it
overshoots the ideal weight values and bounces around the minimum instead
of settling into it. This instability is reflected in Config C having the
lowest accuracy of the three (96.5%) despite reaching a low loss value
early - a lower loss number doesn't always mean a more reliable model if
the training itself was unstable.

## Conclusion
- **Hidden layer size**: Increasing capacity gave a small, real improvement,
  with diminishing returns expected on a dataset this simple.
- **Learning rate**: Faster is not always better. A very high learning rate
  found a low loss quickly but at the cost of stability and final accuracy.
  The baseline learning rate (0.1) gave the best balance of stable training
  and strong accuracy relative to its simplicity.
- **Best overall config**: Config B (32 hidden units, LR 0.1) achieved the
  highest accuracy and F1 score, suggesting that increasing model capacity
  is currently a safer lever to pull than increasing the learning rate for
  this dataset.

## Cross-validation with NumPy Implementation
The from-scratch NumPy implementation (Part 1) achieved 95.0% accuracy on
the same dataset using a comparable architecture (hidden_size=8, ReLU,
LR=0.1), closely matching Config A's 98.0% here. The PyTorch version
performed slightly better, likely due to differences in weight
initialization and the use of mini-batch training (batch_size=32) via
DataLoader, versus full-batch gradient descent in the NumPy version. This
consistency between both implementations is a good sanity check that the
backpropagation math implemented by hand in Part 1 is correct.
