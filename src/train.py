# """
# Main Training Script
# Entry point for training neural networks with command-line arguments
# """

# import argparse

# def parse_arguments():
#     """
#     Parse command-line arguments.
    
#     TODO: Implement argparse with the following arguments:
#     - dataset: 'mnist' or 'fashion_mnist'
#     - epochs: Number of training epochs
#     - batch_size: Mini-batch size
#     - learning_rate: Learning rate for optimizer
#     - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
#     - hidden_layers: List of hidden layer sizes
#     - num_neurons: Number of neurons in hidden layers
#     - activation: Activation function ('relu', 'sigmoid', 'tanh')
#     - loss: Loss function ('cross_entropy', 'mse')
#     - weight_init: Weight initialization method
#     - wandb_project: W&B project name
#     - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
#     """
#     parser = argparse.ArgumentParser(description='Train a neural network')
    
#     return parser.parse_args()


# def main():
#     """
#     Main training function.
#     """
#     args = parse_arguments()
    
#     print("Training complete!")


# if __name__ == '__main__':
#     main()


"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os
import pickle
import wandb

# Import your custom modules
from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data, pre_processing_data

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Dataset and Training Hyperparameters
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help="Dataset to train on")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini-batch size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help="Optimizer choice")
    
    # Network Architecture
    # Note: We treat hidden_layers as the integer count, and num_neurons as the list of sizes
    parser.add_argument('--hidden_layers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('--num_neurons', type=int, nargs='+', default=[128], help="List of hidden layer sizes (space-separated)")
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help="Activation function")
    parser.add_argument('--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help="Loss function")
    parser.add_argument('--weight_init', type=str, choices=['random', 'xavier'], default='xavier', help="Weight initialization method")
    
    # Tracking and Saving
    parser.add_argument('--wandb_project', type=str, default='mlp-from-scratch', help="W&B project name")
    parser.add_argument('--model_save_path', type=str, default='models/trained_weights.pkl', help="Relative path to save trained model")
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    # 1. Parse the arguments from the terminal
    args = parse_arguments()
    
    # Validation check: Ensure the list of neuron sizes matches the number of hidden layers
    if len(args.num_neurons) != args.hidden_layers:
        raise ValueError(f"Error: --hidden_layers is {args.hidden_layers}, but --num_neurons has {len(args.num_neurons)} values.")

    # 2. Initialize Weights & Biases (wandb)
    print(f"Initializing W&B Project: {args.wandb_project}...")
    wandb.init(project=args.wandb_project, config=vars(args))

    # 3. Load and Preprocess Data
    print(f"Loading and preprocessing {args.dataset} dataset...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(args.dataset)
    
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)
    
    # 4. Build the Neural Network
    print(f"Building network with {args.hidden_layers} hidden layer(s)...")
    model = NeuralNetwork(cli_args=args)
    
    # 5. Execute Training Loop
    print("Starting training...")
    list_of_epoch_loss = model.train(
        X_train=X_train, 
        y_train=y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        X_val=X_test, 
        y_val=y_test
    )
    
    # 6. Save the Trained Model safely
    print(f"Saving model to {args.model_save_path}...")
    
    # Create the folder directory (e.g., 'models/') if it doesn't exist yet
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Dump the fully trained memory banks (weights and biases) to disk
    with open(args.model_save_path, 'wb') as f:
        pickle.dump(model.layers, f)
        
    # Close the wandb tracking instance
    wandb.finish()
    
    print("Training complete!")

if __name__ == '__main__':
    main()