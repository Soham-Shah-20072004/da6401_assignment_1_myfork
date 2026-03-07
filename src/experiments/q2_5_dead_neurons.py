"""
Q2.5 The "Dead Neuron" Investigation (6 Marks)
Using ReLU activation and a high learning rate (e.g., 0.1), monitors the activations
of hidden layers to find a run where validation accuracy plateaus early.
Identifies 'dead neurons' (neurons that output zero for all inputs) and compares
with a Tanh run to explain the difference in convergence based on gradients.
"""

import sys
import argparse
import numpy as np
import wandb
import os

# FORCE OFFLINE MODE to prevent Windows ConnectionResetError from killing the loop
os.environ["WANDB_MODE"] = "offline"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, pre_processing_data

def calculate_dead_neurons(model, X_val):
    """
    Passes the entire validation set through the model, and checks
    how many neurons in the hidden layers ALWAYS output 0.0 for every single image.
    Returns: % of dead neurons.
    """
    # Run a forward pass over all validation images
    _ = model.forward(X_val)
    
    total_hidden_neurons = 0
    total_dead_neurons = 0
    
    # Check all layers EXCEPT the last one (output layer)
    for layer in model.layers[:-1]:
        # layer.output contains the activations after applying the activation function
        # Shape of layer.output is (num_neurons, num_samples)
        activations = layer.output
        
        # A neuron is 'dead' if the sum of its absolute activations across all samples is exactly 0
        neuron_activity = np.sum(np.abs(activations), axis=1)
        
        # Count neurons where activity is 0
        dead_count = np.sum(neuron_activity == 0)
        
        total_dead_neurons += dead_count
        total_hidden_neurons += activations.shape[0]
        
    return (total_dead_neurons / total_hidden_neurons) * 100.0, np.concatenate([layer.output.flatten() for layer in model.layers[:-1]])

def run_dead_neuron_experiment(activation, X_train, y_train, X_val, y_val):
    print(f"\n--- Running Experiment: {activation.upper()} with High Learning Rate ---")
    
    wandb.init(
        project="da6401_assignment_1_myfork-src",
        name=f"q2.5-dead-neurons-{activation}",
        group="q2_5_dead_neurons",
        job_type="activation_comparison"
    )
    
    # We use a HIGH learning rate (0.1) with a simple optimizer (SGD) 
    # to specifically trigger permanent dead neurons in ReLU and an early plateau
    learning_rate = 0.1  # Matches assignment suggestion
    
    args = argparse.Namespace(
        dataset='mnist', epochs=10, batch_size=128, 
        optimizer='sgd', learning_rate=learning_rate, num_layers=3, 
        hidden_size=[128, 128, 128], activation=activation, loss='cross_entropy', 
        weight_init='random', weight_decay=0.0, wandb_project="da6401_assignment_1_myfork-src"
    )
    
    model = NeuralNetwork(cli_args=args)
    
    # Custom training loop so we can measure dead neurons EVERY epoch
    print(f"Tracking Dead Neurons over {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # 1. Train the model for one epoch
        model.train(X_train, y_train, epochs=1, batch_size=args.batch_size, X_val=X_val, y_val=y_val)
        
        # 2. Grab a subset of validation images to prevent memory crashes on Windows
        subset_idx = np.random.choice(X_val.shape[1], 1000, replace=False)
        X_val_sub = X_val[:, subset_idx]
        y_val_sub = y_val[:, subset_idx]
        
        val_logits = model.forward(X_val_sub)
        val_preds = np.argmax(val_logits, axis=0)
        true_labels = np.argmax(y_val_sub, axis=0)
        val_acc = np.mean(val_preds == true_labels)
        
        # 3. Calculate % of Dead Neurons and grab raw activations for histogram (on subset)
        dead_percentage, flat_activations = calculate_dead_neurons(model, X_val_sub)
        
        # We also want to look at the gradients during training to explain convergence
        # Let's peek at the gradients right after a training forward+backward pass 
        # (We grab a random batch from training set just to get representational gradients)
        idx = np.random.choice(X_train.shape[1], args.batch_size, replace=False)
        dummy_X, dummy_y = X_train[:, idx], y_train[:, idx]
        model.backward(dummy_y, model.forward(dummy_X))
        flat_gradients = np.concatenate([layer.grad_W.flatten() for layer in model.layers[:-1]])
        
        # 4. Log our custom metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "custom_val_accuracy": val_acc,
            "dead_neuron_percentage": dead_percentage,
            "activation_distribution": wandb.Histogram(flat_activations),
            "gradient_distribution": wandb.Histogram(flat_gradients)
        })
        print(f"  --> Epoch {epoch+1}: Val Acc = {val_acc:.4f}, Dead Neurons = {dead_percentage:.2f}%")
        
    wandb.finish()

def main():
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data('mnist')
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)
    
    # Run Both
    run_dead_neuron_experiment('relu', X_train, y_train, X_test, y_test)
    run_dead_neuron_experiment('tanh', X_train, y_train, X_test, y_test)
    
    print("\n✅ Finished Dead Neurons experiment!")

if __name__ == "__main__":
    main()
