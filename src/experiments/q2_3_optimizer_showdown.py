"""
Q2.3 Optimizer Showdown (5 Marks)
Compares 'sgd', 'momentum', 'nag', and 'rmsprop' on a fixed architecture.
Groups them in W&B to automatically generate comparison plots.
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

def run_optimizer(opt_name, X_train, y_train, X_test, y_test):
    print(f"\n--- Training with Optimizer: {opt_name.upper()} ---")
    
    # We initialize wandb here so we can pass it to NeuralNetwork properly
    wandb.init(
        project="da6401_assignment_1_myfork-src",
        name=f"q2.3-opt-{opt_name}",
        group="q2_3_optimizer_showdown",
        job_type="optimizer_comparison"
    )
    
    args = argparse.Namespace(
        dataset='mnist', epochs=10, batch_size=64, 
        optimizer=opt_name, learning_rate=0.001 if opt_name in ['adam','rmsprop','nadam'] else 0.01, 
        num_layers=3, hidden_size=[128, 128, 128], activation='relu', loss='cross_entropy', 
        weight_init='xavier', weight_decay=0.0, wandb_project="da6401_assignment_1_myfork-src"
    )
    
    model = NeuralNetwork(cli_args=args)
    model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, X_val=X_test, y_val=y_test)
    wandb.finish()

def main():
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data('mnist')
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)
    
    optimizers_to_test = ['sgd', 'momentum', 'nag', 'rmsprop']
    for opt in optimizers_to_test:
        run_optimizer(opt, X_train, y_train, X_test, y_test)
        
    print("\n✅ All optimizers trained! Check the 'q2_3_optimizer_showdown' group in your W&B project.")

if __name__ == "__main__":
    main()
