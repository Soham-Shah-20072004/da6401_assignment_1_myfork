"""
Question 2.6: Loss Function Comparison
Compares Mean Squared Error (MSE) and Cross-Entropy loss for multi-class classification using the same architecture and learning rate.
"""

import sys
import argparse
import os
import wandb


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, pre_processing_data

def run_loss_experiment(loss_name, X_train, y_train, X_test, y_test):
    print(f"\n--- Training with Loss Function: {loss_name.upper()} ---")
    
    # Same architecture and learning rate for both
    args = argparse.Namespace(
        dataset='mnist', epochs=10, batch_size=64, 
        optimizer='sgd', learning_rate=0.01, 
        num_layers=3, hidden_size=[64, 64, 64], activation='relu', loss=loss_name, 
        weight_init='xavier', weight_decay=0.0, wandb_project="da6401_assignment_1_myfork-src"
    )
    
    wandb.init(
        project=args.wandb_project,
        name=f"q2.6-loss-{loss_name}",
        group="q2_6_loss_comparison",
        job_type="loss_comparison",
        config=vars(args)
    )
    
    model = NeuralNetwork(cli_args=args)
    
    # Monkey-patch the evaluate method for THIS script only to avoid OOM ArrayMemoryError
    # when processing the entire validation set gradient at once
    original_evaluate = model.evaluate
    def memory_safe_evaluate(X, y, verbose=True):
        import numpy as np
        from sklearn.metrics import f1_score
        from ann import objective_functions
        
        chunk_size = 500
        total_correct = 0
        total_loss = 0.0
        all_pred = []
        all_true = []
        
        for i in range(0, X.shape[1], chunk_size):
            X_chunk = X[:, i:i+chunk_size]
            y_chunk = y[:, i:i+chunk_size]
            
            output_chunk = model.forward(X_chunk)
            softmax_probs_chunk = model.layers[-1].output
            
            pred_indices = np.argmax(output_chunk, axis=0)
            true_indices = np.argmax(y_chunk, axis=0)
            
            total_correct += np.sum(pred_indices == true_indices)
            all_pred.extend(pred_indices)
            all_true.extend(true_indices)
            
            if model.loss_type == 'cross_entropy':
                chunk_loss = np.sum(objective_functions.categorical_cross_entropy(y_chunk, softmax_probs_chunk))
            else:
                chunk_loss = np.sum(objective_functions.mse(y_chunk, softmax_probs_chunk))
            total_loss += chunk_loss
            
        accuracy = total_correct / y.shape[1]
        loss = total_loss / y.shape[1]
        f1 = f1_score(all_true, all_pred, average='macro')
        
        if verbose:
            print(f"Evaluation Accuracy: {accuracy}")
            print(f"Evaluation F1 Score: {f1}")
            print(f"Evaluation Loss: {loss}")
            
        return accuracy, loss, f1
    
    model.evaluate = memory_safe_evaluate
    
    model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, X_val=X_test, y_val=y_test)
    wandb.finish()

def main():
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data('mnist')
    
    # Using full dataset now that memory constraints are handled during evaluation    
    X_train, y_train = pre_processing_data(X_train_raw, y_train_raw)
    X_test, y_test = pre_processing_data(X_test_raw, y_test_raw)
    
    loss_functions_to_test = ['mse', 'cross_entropy']
    for loss in loss_functions_to_test:
        run_loss_experiment(loss, X_train, y_train, X_test, y_test)
        
    print("\n[DONE] Both models trained! Check the 'q2_6_loss_comparison' group in your W&B project.")

if __name__ == "__main__":
    main()
