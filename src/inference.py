"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as cm
from ann.neural_network import NeuralNetwork
import ann.objective_functions as obj_funcs
from ann.activations import softmax
import utils.data_loader as data_loader


def parse_arguments():
    """
    Parse command-line arguments for inference.
    CLI matches train.py exactly (as required by the assignment), plus --model_path for loading saved weights.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    # Dataset and Training Hyperparameters (same flags as train.py)
    parser.add_argument('-d','--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help="Dataset to evaluate on")
    parser.add_argument('-e','--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('-b','--batch_size', type=int, default=64, help="Mini-batch size")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument('-wd','--weight_decay', type=float, default=0.0005, help="Weight decay")
    parser.add_argument('-o','--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='momentum', help="Optimizer choice")
    
    # Network Architecture(same cli flags as train.py)
    parser.add_argument('-nhl','--num_layers', type=int, default=3, help="Number of hidden layers")
    parser.add_argument('-sz','--hidden_size', type=int, nargs='+', default=[128], help="List of hidden layer sizes (space-separated)")
    parser.add_argument('-a','--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help="Activation function")
    parser.add_argument('-l','--loss', type=str, choices=['cross_entropy', 'mse', 'mean_squared_error'], default='cross_entropy', help="Loss function")
    parser.add_argument('-w_i','--weight_init', type=str, choices=['random', 'xavier', 'zeros'], default='xavier', help="Weight initialization method")
    
    # Tracking and Saving(same cli flags as train.py)
    parser.add_argument('-w_p','--wandb_project', type=str, default='da6401_assignment_1_myfork-src', help="W&B project name")
    parser.add_argument('--model_save_path', type=str, default='src/best_model.npy', help="Relative path to save trained model")
    
    # Inference-specific argument
    parser.add_argument('--model_path', type=str, default='src/best_model.npy', help="Relative path to saved model weights (e.g., src/best_model.npy)")

    return parser.parse_args()

def load_model(model_path, args):
    """
    Load trained model from disk.
    Uses the same CLI args structure as train.py to rebuild the network shell,
    then injects the saved weights via deserialization.
    """
    # build the empty shell network using CLI args (same structure as train.py)
    model = NeuralNetwork(args)

    # load the saved weights and inject into this empty placeholder model
    # this is DESERIALIZATION - we load serialised weights from .npy file
    saved_weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(saved_weights)
    # now this model is a smart model with trained weights
    return model


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    Returns Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    # Forward pass through the model - returns raw logits(pre-softmax)
    logits = model.forward(X_test)
    
    # Apply softmax to get probabilities for loss computation
    probs = softmax(logits)
    
    predictions = np.argmax(logits, axis=0)  # argmax on logits gives same result as on softmax probs
    if y_test.ndim == 2 and y_test.shape[0] != 10:
        y_test_col = y_test.T
    else:
        y_test_col = y_test
    
    # compute loss using softmax probabilities
    # objective_functions returns individual losses per sample (shape N,), hence we average them
    loss = np.mean(obj_funcs.categorical_cross_entropy(y_test, probs)) 

    # Calculate Sklearn Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    conf_mat = cm(true_labels, predictions)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_mat
    }

def main():
    """
    Main inference function.

    Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    # Normalize loss name - support both 'mse' and 'mean_squared_error'
    if args.loss == 'mean_squared_error':
        args.loss = 'mse'
    
    # W&B Sweep sends 'hidden_size' as a single scalar integer. We must cast it to a list
    # of that integer repeated `num_layers` times so the NeuralNetwork can parse it.
    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
    
    print(f"Loading {args.dataset} test data...")
    # Load test data based on the specified dataset
    _,_,X_test_raw,y_test_raw = data_loader.load_data(args.dataset)
    X_test, y_test = data_loader.pre_processing_data(X_test_raw, y_test_raw)

    # load model - loads the weights and injects into empty shell, returns the smart model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args)
    
    # evaluate the model
    print("Evaluating model and calculating metrics...")
    results_metrics_dict = evaluate_model(model, X_test, y_test)

    # Print the terminal report
    print("\n" + "="*40)
    print("INFERENCE EVALUATION REPORT")
    print("="*40)
    print(f"Test Loss:      {results_metrics_dict['loss']:.4f}")
    print(f"Test Accuracy:  {results_metrics_dict['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:{results_metrics_dict['precision'] * 100:.2f}%")
    print(f"Macro Recall:   {results_metrics_dict['recall'] * 100:.2f}%")
    print(f"Macro F1 Score: {results_metrics_dict['f1'] * 100:.2f}%")
    print("="*40 + "\n")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
