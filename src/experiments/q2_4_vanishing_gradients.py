"""
Q2.4 Vanishing Gradients (5 Marks)
Measures the gradient norm of the FIRST hidden layer and LAST hidden layer
for both Sigmoid and ReLU activations across varying network depths.
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
from utils.data_loader import load_data, pre_processing_data, batch_generator

def run_vanishing_grad_experiment(activation, depth, X_train, y_train):
    print(f"\n--- Running {activation.upper()} with {depth} hidden layers ---")
    
    wandb.init(
        project="da6401_assignment_1_myfork-src",
        name=f"q2.4-{activation}-depth{depth}",
        group="q2_4_vanishing_gradients",
        job_type="vanishing_grad_check"
    )
    
    hidden_sizes = [64] * depth
    args = argparse.Namespace(
        dataset='mnist', epochs=1, batch_size=64, 
        optimizer='rmsprop', learning_rate=0.001, num_layers=depth, 
        hidden_size=hidden_sizes, activation=activation, loss='cross_entropy', 
        weight_init='xavier', weight_decay=0.0, wandb_project="da6401_assignment_1_myfork-src"
    )
    
    model = NeuralNetwork(cli_args=args)
    
    batches_to_run = 100
    batch_count = 0
    
    for X_batch, y_batch in batch_generator(X_train, y_train, args.batch_size):
        if batch_count >= batches_to_run:
            break
            
        batch_pred = model.forward(X_batch)
        model.backward(y_batch, batch_pred)
        
        # Measure norms
        first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)
        last_hidden_layer_grad_norm = np.linalg.norm(model.layers[-2].grad_W) # -1 is output layer, -2 is last hidden 
        
        wandb.log({
            "batch": batch_count,
            "first_hidden_layer_grad_norm": first_layer_grad_norm,
            "last_hidden_layer_grad_norm": last_hidden_layer_grad_norm,
            "ratio_(first_div_last)": first_layer_grad_norm / (last_hidden_layer_grad_norm + 1e-8)
        })
        
        model.update_weights()
        batch_count += 1
        
    wandb.finish()

def main():
    print("Loading data...")
    X_train_raw, y_train_raw, _, _ = load_data('mnist')
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    
    for activation in ['sigmoid', 'relu']:
        for depth in [3, 5]:
            run_vanishing_grad_experiment(activation, depth, X_train, y_train)
            
    print("\n✅ Finished Vanishing Gradients experiment!")

if __name__ == "__main__":
    main()
