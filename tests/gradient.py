import numpy as np
import sys
import os

sys.path.append(os.path.abspath("src"))

from ann.neural_network import NeuralNetwork
from ann.neural_layer import Layer

# -------------------------
# Fake CLI args
# -------------------------
class Args:
    pass

args = Args()
args.input_size = 4
args.num_layers = 1
args.hidden_size = [5]
args.output_size = 3
args.activation = "relu"
args.weight_init = "random"
args.loss = "cross_entropy"
args.learning_rate = 0.01
args.weight_decay = 0.0
args.optimizer = "sgd"

model = NeuralNetwork(args)

np.random.seed(42)

for layer in model.layers:
    if isinstance(layer, Layer):
        layer.W = np.random.randn(*layer.W.shape) * 0.1
        layer.bias = np.random.randn(*layer.bias.shape) * 0.1


X = np.random.randn(784, 2)
y = np.zeros((10, 2))
y[np.random.randint(0, 10, 2), np.arange(2)] = 1

# -------------------------
# Forward + backward (analytical)
# -------------------------
y_pred = model.forward(X)
model.backward(y, y_pred)

# Pick first Linear layer
layer = None
for l in model.layers:
    if isinstance(l, Layer):
        layer = l
        break

analytical_grad = layer.grad_W.copy()

# -------------------------
# Numerical gradient
# -------------------------
epsilon = 1e-5
numerical_grad = np.zeros_like(layer.W)

from ann import objective_functions
from ann.activations import softmax

for i in range(layer.W.shape[0]):
    for j in range(layer.W.shape[1]):
        original = layer.W[i, j]

        layer.W[i, j] = original + epsilon
        plus_loss = np.mean(objective_functions.categorical_cross_entropy(y, softmax(model.forward(X))))

        layer.W[i, j] = original - epsilon
        minus_loss = np.mean(objective_functions.categorical_cross_entropy(y, softmax(model.forward(X))))

        numerical_grad[i, j] = (plus_loss - minus_loss) / (2 * epsilon)

        layer.W[i, j] = original # restore

# -------------------------
# Relative error
# -------------------------
num = np.linalg.norm(analytical_grad - numerical_grad)
den = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
rel_error = num / den

print("Relative Error:", rel_error)

if rel_error < 1e-7:
    print("PASS: Gradients correct")
else:
    print("FAIL: Gradients incorrect")
