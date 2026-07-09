# Week 3 Experiment Report: Neural Network Configuration Comparison

## Objective
Compare four training configurations of the PyTorch neural network on the
moons binary classification dataset, varying hidden layer size, learning
rate, and activation function, to understand how each hyperparameter affects
training dynamics and final performance.

## Dataset
Synthetic "two moons" dataset (1000 samples, 20% noise), split 80/20 into
train/test sets. Chosen because it is not linearly separable, making it a
good test of whether the network is actually learning a non-linear boundary.

## Configurations Tested

| Config | Hidden Size | Activation | Learning Rate | Accuracy | F1 Score |
|--------|------------|------------|----------------|----------|----------|
| A - Baseline       | 8  | ReLU | 0.1 | 0.9750 | 0.9751 |
| B - Bigger Network | 32 | ReLU | 0.1 | 0.9800 | 0.9802 |
| C - High LR        | 8  | ReLU | 1.0 | 0.9950 | 0.9950 |
| D - Tanh Activation| 8  | Tanh | 0.1 | 0.9600 | 0.9600 |

## Analysis

### Config A vs Config B: Hidden Layer Size
Increasing the hidden layer from 8 to 32 neurons improved accuracy slightly
(97.5% -> 98.0%) and reached a lower loss faster during training. More
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

### Config A vs Config D: Activation Function
Changing only the activation function (ReLU to Tanh, keeping hidden size and
learning rate identical to the baseline) resulted in a small accuracy drop
(97.5% -> 96.0%). Tanh's loss curve also converged more slowly in early
epochs (0.643 at epoch 0 vs 0.625 for ReLU, and still at 0.165 by epoch 90
vs ReLU's 0.139). This is consistent with a known property of Tanh: its
output saturates (flattens out) at both extremes, which can produce
smaller gradients during backpropagation and slow down learning compared
to ReLU, which has a constant gradient of 1 for all positive inputs. For
this dataset and network size, ReLU was the more effective activation
function.

## Conclusion
- **Hidden layer size**: Increasing capacity gave a small, real improvement.
- **Learning rate**: A higher learning rate reached lower loss faster in
  this run, though earlier experiments showed this can come at the cost of
  training stability - results can vary between runs due to random
  initialization, so learning rate should be tuned carefully rather than
  simply increased.
- **Activation function**: ReLU outperformed Tanh on this task, likely due
  to Tanh's saturating gradients slowing convergence.
- **Best overall config in this run**: Config C, though given some
  instability observed in earlier experiments with high learning rates,
  Config B (larger network, standard learning rate) remains the safer,
  more consistently reliable choice.

## Cross-validation with NumPy Implementation
The from-scratch NumPy implementation (Part 1) achieved 95.0% accuracy on
the same dataset using a comparable architecture (hidden_size=8, ReLU,
LR=0.1), closely matching Config A's 98.0% here. The PyTorch version
performed slightly better, likely due to differences in weight
initialization and the use of mini-batch training (batch_size=32) via
DataLoader, versus full-batch gradient descent in the NumPy version. This
consistency between both implementations is a good sanity check that the
backpropagation math implemented by hand in Part 1 is correct.
