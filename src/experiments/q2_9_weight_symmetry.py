"""
Q2.9 Weight Init & Symmetry (7 Marks)
Compares 'zeros' vs 'xavier' initialization.
Tracks the gradient norms of 5 specific neurons in the first hidden layer
to prove that zero-initialization fails to break symmetry.
"""

import sys
import argparse
import numpy as np
import wandb
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, pre_processing_data, batch_generator

def run_symmetry_experiment(weight_init, X_train, y_train):
    print(f"\n--- Running Experiment: {weight_init.upper()} Initialization ---")
    
    wandb.init(
        project="da6401_assignment_1_myfork-src",
        name=f"q2.9-symmetry-{weight_init}",
        group="q2_9_weight_symmetry",
        job_type="symmetry_check"
    )
    
    # Create fake CLI args to pass to NeuralNetwork
    args = argparse.Namespace(
        dataset='mnist', epochs=1, batch_size=64, 
        optimizer='sgd', learning_rate=0.01, num_layers=2, 
        hidden_size=[128, 128], activation='relu', loss='cross_entropy', 
        weight_init=weight_init, weight_decay=0.0, wandb_project="da6401_assignment_1_myfork-src"
    )
    
    model = NeuralNetwork(cli_args=args)
    
    # We don't use model.train() because we want to intercept gradients *during* the batch loop
    batches_to_run = 50
    batch_count = 0
    
    for X_batch, y_batch in batch_generator(X_train, y_train, args.batch_size):
        if batch_count >= batches_to_run:
            break
            
        # Forward pass
        batch_pred = model.forward(X_batch)
        
        # Backward pass (computes gradients)
        model.backward(y_batch, batch_pred)
        
        # Now intercept the gradients of the first hidden layer BEFORE weights are updated
        first_hidden_layer = model.layers[0]
        grad_W = first_hidden_layer.grad_W  # Shape: (input_dim, num_neurons) = (784, 128)
        
        # We track the gradient norm of the first 5 neurons
        # For a single neuron 'j', its weights are grad_W[:, j]
        logs = {"batch": batch_count}
        for neuron_idx in range(5):
            neuron_grad = grad_W[neuron_idx, :]
            grad_norm = np.linalg.norm(neuron_grad)
            logs[f"neuron_{neuron_idx}_grad_norm"] = grad_norm
            
        wandb.log(logs)
        
        # Update weights
        model.update_weights()
        batch_count += 1
        
    wandb.finish()

def main():
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data('mnist')
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    
    run_symmetry_experiment('zeros', X_train, y_train)
    run_symmetry_experiment('xavier', X_train, y_train)

if __name__ == "__main__":
    main()
