

"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os
import pickle
import wandb
import numpy as np

# Import network module
from ann.neural_network import NeuralNetwork
# importing helper functiond from utilities
from utils.data_loader import load_data, pre_processing_data

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Dataset and Training Hyperparameters
    parser.add_argument('-d','--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help="Dataset to train on")
    parser.add_argument('-e','--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('-b','--batch_size', type=int, default=64, help="Mini-batch size")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument('-wd','--weight_decay', type=float, default=0.0005, help="Weight decay")
    parser.add_argument('-o','--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='momentum', help="Optimizer choice")
    
    # Network Architecture
    parser.add_argument('-nhl','--num_layers', type=int, default=3, help="Number of hidden layers")
    parser.add_argument('-sz','--hidden_size', type=int,nargs='+', default=[128], help="List of hidden layer sizes (space-separated)")
    parser.add_argument('-a','--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help="Activation function")
    parser.add_argument('-l','--loss', type=str, choices=['cross_entropy', 'mse', 'mean_squared_error'], default='cross_entropy', help="Loss function")
    parser.add_argument('-w_i','--weight_init', type=str, choices=['random', 'xavier', 'zeros'], default='xavier', help="Weight initialization method")
    
    # Tracking and Saving
    parser.add_argument('-w_p','--wandb_project', type=str, default='da6401_assignment_1_myfork-src', help="W&B project name")
    parser.add_argument('--model_save_path', type=str, default='src/trained_model.npy', help="Relative path to save trained model (use src/best_model.npy to update the best model)")
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    # loss
    if args.loss == 'mean_squared_error':
        args.loss = 'mse'
    
    # Validation check: Ensure the list of neuron sizes matches the number of hidden layers

    # W&B Sweep sends 'hidden_size' as a single scalar integer. We must cast it to a list
    # of that integer repeated `num_layers` times so the NeuralNetwork can parse it.
    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
        print(f"Auto-expanded hidden_size to {args.hidden_size} to match num_layers={args.num_layers}")
        
    if len(args.hidden_size) != args.num_layers:
        raise ValueError(f"Error: --num_layers is {args.num_layers}, but --hidden_size has {len(args.hidden_size)} values.")

    # Load and Preprocess Data
    print(f"Loading and preprocessing {args.dataset} dataset...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(args.dataset)
    
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)

    # Initialize Weights & Biases (wandb)
    print(f"Initializing W&B Project: {args.wandb_project}...")
    # Fall back to offline mode when no API key is present (e.g., autograder environment)
    # This prevents wandb.init() from hanging on network calls without credentials.
    import os
    if not os.environ.get('WANDB_API_KEY') and not os.environ.get('WANDB_MODE'):
        os.environ['WANDB_MODE'] = 'offline'
    try:
        wandb.init(project=args.wandb_project, config=vars(args))
    except Exception as e:
        print(f"W&B init failed ({e}), falling back to disabled mode.")
        wandb.init(project=args.wandb_project, config=vars(args), mode='disabled')
    
    # Build the Neural Network
    print(f"Building network with {args.num_layers} hidden layer(s)...")
    model = NeuralNetwork(cli_args=args)
    
    # Execute Training Loop
    print("Starting training...")
    list_of_epoch_loss = model.train(
        X_train=X_train, 
        y_train=y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        X_val=X_test, 
        y_val=y_test
    )
    
    # Save the Trained Model
    print(f"Saving model to {args.model_save_path}...")
    
    # Create the folder directory (e.g., 'models/') if it doesn't exist yet
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # SAVE THE MODEL WEIGHTS AS BEST WEIGHTS AND BEST MODEL.npy file
    # this is SERIALIZATION . in inference, we will deserialize this best model file to inject the weights. for that we have defined method of network named set_weights().
    print(f"Extracting weights and saving serialised weights to src/best_model.npy...")
    best_weights = model.get_weights()
    np.save(args.model_save_path, best_weights) # serialised weights, saved as .npy file, instead of prev .pkl file
    
    print(f"Saving hyperparameter config to src/best_config.json...")
    import json
    with open('src/best_config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # Close the wandb tracking instance
    wandb.finish()
    
    print("Training complete!")

if __name__ == '__main__':
    main()
