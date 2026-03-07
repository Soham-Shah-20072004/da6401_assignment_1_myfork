import sys
import os

# Add the 'src' directory to the Python path so local modules import correctly
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ann.neural_network import NeuralNetwork
from ann import objective_functions
from ann.activations import softmax
from utils import data_loader

def test_combination(opt_name, loss_fn):
    class DummyArgs:
        num_layers = 1
        hidden_size = [64]
        activation = 'relu'
        loss = loss_fn
        optimizer = opt_name
        learning_rate = 0.001
        weight_decay = 0.0
        weight_init = 'xavier'
        
    args = DummyArgs()
    model = NeuralNetwork(args)
    
    np.random.seed(42)  
    X_dummy = np.random.rand(784, 5)  
    y_dummy_raw = np.array([0, 1, 2, 3, 4]) 
    ten_ten_identity = np.eye(10)
    y_dummy = ten_ten_identity[y_dummy_raw].T 
    
    initial_logits = model.forward(X_dummy)
    initial_probs = softmax(initial_logits)
    if loss_fn == 'cross_entropy':
        initial_loss = np.mean(objective_functions.categorical_cross_entropy(y_dummy, initial_probs))
    else:
        initial_loss = np.mean(objective_functions.mse(y_dummy, initial_probs))
        
    for epoch in range(50):
        pred = model.forward(X_dummy)
        model.backward(y_dummy, pred)
        model.update_weights()
        
    final_logits = model.forward(X_dummy)
    final_probs = softmax(final_logits)
    if loss_fn == 'cross_entropy':
        final_loss = np.mean(objective_functions.categorical_cross_entropy(y_dummy, final_probs))
    else:
        final_loss = np.mean(objective_functions.mse(y_dummy, final_probs))
        
    return initial_loss, final_loss

def run_all_sanity_tests():
    optimizers = ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
    losses = ['cross_entropy', 'mse']
    
    print("=== Comprehensive Math Sanity Check ===")
    print("Testing 12 combinations of Optimizers and Loss Functions...")
    all_passed = True
    for opt in optimizers:
        for loss in losses:
            try:
                initial_loss, final_loss = test_combination(opt, loss)
                if final_loss < initial_loss:
                    print(f"[PASS] Optimizer: {opt.ljust(10)} | Loss: {loss.ljust(15)} (Loss {initial_loss:.4f} -> {final_loss:.4f})")
                else:
                    print(f"[FAIL] Optimizer: {opt.ljust(10)} | Loss: {loss.ljust(15)} (Loss {initial_loss:.4f} -> {final_loss:.4f})")
                    all_passed = False
            except Exception as e:
                print(f"[ERROR] Optimizer: {opt.ljust(10)} | Loss: {loss.ljust(15)} Failed with error: {str(e)}")
                all_passed = False
                
    if all_passed:
        print("\nALL CONFIGURATIONS ARE MATHEMATICALLY SOUND AND LEARN SUCCESSFULLY!")
    else:
        print("\nSOME CONFIGURATIONS FAILED. Please check the logs.")

if __name__ == "__main__":
    run_all_sanity_tests()
