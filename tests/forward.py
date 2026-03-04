import numpy as np
import sys
import os

sys.path.append(os.path.abspath("src"))

from ann.neural_network import NeuralNetwork
from ann.neural_layer import Layer

# -----------------------------
# STEP 1: Create fake CLI args
# -----------------------------
class Args:
    pass

args = Args()
args.input_size = 4
args.num_layers = 1
args.hidden_size = [3]
args.output_size = 2
args.activation = "relu"
args.weight_init = "random"
args.loss = "cross_entropy"
args.learning_rate = 0.01
args.weight_decay = 0.0
args.optimizer = "rmsprop"

# -----------------------------
# STEP 2: Build model
# -----------------------------
model = NeuralNetwork(args)

# -----------------------------
# STEP 3: Set FIXED weights
# -----------------------------
for layer in model.layers:
    if isinstance(layer, Layer):
        layer.W = np.ones_like(layer.W) * 0.5
        layer.bias = np.ones_like(layer.bias) * 0.1

# -----------------------------
# STEP 4: Create fixed input
# -----------------------------
# Hardcode determinism so it passes every time
np.random.seed(42)
X = np.random.randn(784, 2)  # MNIST uses 784-dimensional features

# -----------------------------
# STEP 5: Run forward twice
# -----------------------------
out1 = model.forward(X)
out2 = model.forward(X)

# -----------------------------
# STEP 6: Determinism check
# -----------------------------
print("Deterministic:", np.allclose(out1, out2))

# -----------------------------
# STEP 7: Shape check
# -----------------------------
print("Output shape:", out1.shape) # should be (10, 2) since NeuralNetwork hardcodes 10 output classes

# -----------------------------
# STEP 8: Print logits (REFERENCE)
# -----------------------------
np.set_printoptions(precision=6, suppress=True)
print("Logits:")
print(out1)