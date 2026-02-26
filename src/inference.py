"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as cm
from src.ann.neural_network import NeuralNetwork
import src.ann.objective_functions as obj_funcs
import src.utils.data_loader as data_loader


def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    # Core inference arguments
    parser.add_argument('--model_path', type=str, required=True, help="please give relative path to saved model weights (e.g., models/weights.pkl)")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help="choose the Dataset to evaluate on")
    parser.add_argument('--batch_size', type=int, default=32, help="choose the batch size for inference")
    
    # Architecture arguments to rebuild the empty shell
    parser.add_argument('--hidden_layers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('--num_neurons', type=int, nargs='+', default=[128], help="List of hidden layer sizes")
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help="choose activation function used during training")
    
    return parser.parse_args()

def load_model(model_path, args):
    """
    Load trained model from disk.
    """
    # model path is the path to saved weights file(pkl file)
    # Create a dummy object to mimic the terminal args your __init__ expects
    class DummyArgs:
        loss = 'cross_entropy'
        optimizer = 'sgd' # no need, Ignored during inference, but needed to initalize the empty neural network
        num_layers = args.hidden_layers
        hidden_size = args.num_neurons
        activation = args.activation
        weight_init = 'random'
    
    # build the empty shell
    model = NeuralNetwork(DummyArgs())

    # load the saved weights and inject into this empty placeholder model
    with open(model_path, 'rb') as f:
        saved_weights = pkl.load(f)

    model.layers = saved_weights['layers']
    # these saved_weights are simply the weights and biases of each layer
    # fundamentally it could be a layer object (smart/trained layers object list)

    # now this model is a smart model
    return model


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall

    """

    # Forward pass through the model
    logits = model.forward(X_test.T)  # Transpose to match input shape (features, samples)
    predictions = np.argmax(logits, axis=0)  # Get predicted class labels
    true_labels = np.argmax(y_test.T, axis=0)  # Get true class labels from
    loss = obj_funcs.categorical_cross_entropy(y_test.T, logits) # compute loss using the categorical cross entropy function implemented in objective functions file
    

    # 4. Calculate Sklearn Metrics
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

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    print(f"Loading {args.dataset} test data...") # now we start to load the specified dataset
    # Load test data based on the specified dataset
    _,_,X_test_raw,y_test_raw = data_loader.load_data(args.dataset) # this will give us the test data and labels, we need to pre process them before feeding into the model for evaluation
    X_test, y_test = data_loader.pre_processing_data(X_test_raw, y_test_raw)

    # we loaded the dataset, now load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args) # this loads the smart model in model, we need to evaluate this
    
    # we evaluate
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
