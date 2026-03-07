"""
Question 2.8: Error Analysis
Generates a Confusion Matrix and visualizes model failures.
All results are explicitly logged to Weights and Biases (W&B).
"""

import os
import sys
import json
import argparse
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, pre_processing_data

def load_best_model():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'best_config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create namespace object from dict for compatibility with NeuralNetwork initialization
    args = argparse.Namespace(**config_dict)
    
    # Init model
    model = NeuralNetwork(cli_args=args)
    
    # Load weights
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.npy')
    best_weights = np.load(weights_path, allow_pickle=True).item()
    model.set_weights(best_weights)
    
    return model, args

def get_failure_images(X_test_raw, y_true, y_pred, softmax_probs):
    # Find incorrect predictions
    incorrect_indices = np.where(y_true != y_pred)[0]
    
    # Calculate confidence for the INCORRECT prediction
    wrong_confidences = []
    for idx in incorrect_indices:
        pred_class = y_pred[idx]
        confidence = softmax_probs[pred_class, idx]
        wrong_confidences.append((idx, confidence))
        
    # Sort by confidence (highest confidence in WRONG answer first)
    wrong_confidences.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 15 most confidently wrong
    top_failures = wrong_confidences[:15]
    
    wandb_images = []
    for idx, conf in top_failures:
        # Raw image is 784, reshape to 28x28
        img = X_test_raw[idx].reshape(28, 28)
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        caption = f"Pred: {pred_label} ({conf:.2f}) | True: {true_label}"
        
        wandb_images.append(wandb.Image(img, caption=caption))
        
    return wandb_images

def main():
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data('mnist')
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)
    
    print("Loading best model...")
    model, args = load_best_model()
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name="q2.8-error-analysis",
        group="q2_8_error_analysis",
        job_type="evaluation",
        config=vars(args)
    )
    
    print("Running inference on test set...")
    # X_test is (784, 10000), output is (10, 10000)
    output = model.forward(X_test)
    softmax_probs = model.layers[-1].output
    
    y_pred = np.argmax(output, axis=0) # Shape: (10000,)
    y_true = np.argmax(y_test, axis=0) # Shape: (10000,)
    
    # 1. Confusion Matrix
    print("Logging Confusion Matrix to W&B...")
    
    # Log CM to WandB natively
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true, 
            preds=y_pred,
            class_names=[str(i) for i in range(10)]
        )
    })
    
    # 2. Failure Visualization
    print("Logging Model Failures to W&B...")
    
    # Get WandB Images for Failures
    failure_images = get_failure_images(X_test_raw, y_true, y_pred, softmax_probs)
    
    # Log Failures to WandB
    wandb.log({
        "confident_failures": failure_images
    })
    
    wandb.finish()
    print("Done! Artifacts logged to W&B.")

if __name__ == "__main__":
    main()
