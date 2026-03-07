"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
import wandb
from sklearn.metrics import f1_score

from ann import optimizers
from ann.neural_layer import Layer
from ann import objective_functions
from utils.data_loader import batch_generator


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        self.num_of_hidden_layers = cli_args.num_layers
        self.hidden_layers_sizes = cli_args.hidden_size
        self.weight_init_method = cli_args.weight_init
        self.hidden_activation_function = cli_args.activation
        self.loss_type = cli_args.loss
        # 1. Initialize the Optimizer Switchboard
        if cli_args.optimizer == 'sgd':
            self.optimizer = optimizers.SGD(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'momentum':
            self.optimizer = optimizers.Momentum(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'nag':
            self.optimizer = optimizers.NAG(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        elif cli_args.optimizer == 'nadam':
            self.optimizer = optimizers.Nadam(learning_rate=cli_args.learning_rate, weight_decay=cli_args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {cli_args.optimizer}")

        self.input_size = 784  # for MNIST dataset
        self.output_size = 10  # for MNIST dataset

        if isinstance(self.hidden_layers_sizes, int):
            self.hidden_layers_sizes = [self.hidden_layers_sizes] * self.num_of_hidden_layers
        elif len(self.hidden_layers_sizes) == 1 and self.num_of_hidden_layers > 1:
            self.hidden_layers_sizes = self.hidden_layers_sizes * self.num_of_hidden_layers
        elif len(self.hidden_layers_sizes) < self.num_of_hidden_layers:
            # if not enough sizes were provided, padding
            last = self.hidden_layers_sizes[-1]
            self.hidden_layers_sizes = list(self.hidden_layers_sizes) + \
                [last] * (self.num_of_hidden_layers - len(self.hidden_layers_sizes))


        for i in range(self.num_of_hidden_layers):
            layer_type = "hidden"
            if i==0:
                current_input_size = self.input_size
            else:
                current_input_size = self.hidden_layers_sizes[i-1] # for hidden layer i (starting from 0 index), the input size is the layer size of the i-1 th hidden layer
            
            hidden_layer_i = Layer(current_input_size, self.hidden_layers_sizes[i], weight_init=self.weight_init_method, layer_type=layer_type, activation_function=self.hidden_activation_function)
            self.layers.append(hidden_layer_i)

            # so we are done with a list of hidden layers

        # now we add the output layer, which takes input from the last hidden layer and outputs the final output
        output_layer = Layer(input_size=self.hidden_layers_sizes[-1], layer_size=self.output_size, weight_init=self.weight_init_method,layer_type="output")
        self.layers.append(output_layer)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        X = np.array(X)
        row_major = (X.ndim == 2 and
                     X.shape[1] == self.input_size and
                     X.shape[0] != self.input_size)
        if row_major:
            X = X.T  # (N, D) -> (D, N) 
        current_input_signal = X
        for layer in self.layers:
            current_output = layer.forward(current_input_signal)
            current_input_signal = current_output
        logits = self.layers[-1].weighted_sum  # always (classes, N) 

        # Return (N, classes) when caller passes row-major 
        if row_major:
            return logits.T  # (N, classes)
        return logits
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
        """
        # y_pred is logits (pre-softmax). Get softmax probabilities from the output layer's stored activation.

        y_true = np.array(y_true)
        if y_true.ndim == 0:
            # scalar class index → one-hot column vec
            oh = np.zeros((self.output_size, 1))
            oh[int(y_true), 0] = 1.0
            y_true = oh
        elif y_true.ndim == 1:
            if y_true.shape[0] == self.output_size and np.all((y_true == 0) | (y_true == 1)):
                # already a one-hot vector
                y_true = y_true.reshape(self.output_size, 1)
            else:
                # array of integers class index
                N = y_true.shape[0]
                oh = np.zeros((self.output_size, N))
                oh[y_true.astype(int), np.arange(N)] = 1.0
                y_true = oh
        elif y_true.ndim == 2:
            if y_true.shape[0] != self.output_size and y_true.shape[1] == self.output_size:
                y_true = y_true.T 


        softmax_probs = self.layers[-1].output
        
        # Determine the gradient of the loss with respect to the output (dL/da)
        if self.loss_type == 'cross_entropy':
            # The derivative of Categorical Cross Entropy simplifies beautifully with Softmax:
            # dL/dz = softmax_probs - y_true
            # (Note: This is dL/dz directly, so we don't multiply by activation derivative again)
            error_signal = softmax_probs - y_true 
        elif self.loss_type == 'mse':
            # For MSE, dL/da is the typical derivative
            dL_da = objective_functions.mse_derivative(y_true, softmax_probs) 
            
            # Since Softmax output a_i depends on ALL z_j, we must multiply by the Softmax Jacobian:
            # J_ij = a_i * (delta_ij - a_j)
            # Multiplying J by the gradient vector dL/da gives:
            # dL/dz_i = a_i * dL/da_i - a_i * sum(a_j * dL/da_j)
            
            # This computes the exact exact Jacobian-vector product in vectorized numpy for the entire batch
            sum_a_dL_da = np.sum(softmax_probs * dL_da, axis=0, keepdims=True)
            error_signal = (softmax_probs * dL_da) - (softmax_probs * sum_a_dL_da)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        all_grad_W = []  # grad_W each layer,last -->first
        all_grad_b = []  # grad_b each layer,last -->first
        for layer in reversed(self.layers):
            intermed_error = layer.backward(error_signal)  # this will compute and store the graadients and return del.w value which is the input for the next layer in the reverse order
            error_signal = intermed_error
            all_grad_W.append(layer.grad_W)
            all_grad_b.append(layer.grad_b)
            
        # with this we have gradients stored in each layer object where each value in the matrix rep the sum of gradients of Loss wrt to that weight across the batch
        # so we might have to divide by batch size to get the mean gradient for each weight, and then we can use these gradients to update the weights using the optimizer, which will be implemented in the update_weights() method of this class.
        return all_grad_W, all_grad_b


    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update(self.layers)

    # setting the getter method for weights and biases for each layer in form of dictionary   
    def get_weights(self):
        """
        Extract weights and biases from all layers for serialization.(serialization means to convert this object into bytestream. this we do into .npy file or .pkl file)
        Returns a dictionary containing W and bias arrays for each layer.
        """
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            #  flat format with _W / _b suffix
            weights_dict[f'layer_{i}_W'] = layer.W.copy()
            weights_dict[f'layer_{i}_b'] = layer.bias.copy()
            # Alias with bias suffix for backward compatibility
            weights_dict[f'layer_{i}_bias'] = layer.bias.copy()
            # Nested dict 
            weights_dict[f'layer_{i}'] = {'W': layer.W.copy(), 'b': layer.bias.copy()}
        return weights_dict
        
    def set_weights(self, weights_dict):
        """
        Inject deserialized Numpy array weights and biases back into Layer objects.
        """
        def _extract(entry):
            W = entry.get('W', entry.get('weight', entry.get('weights')))
            b = entry.get('b', entry.get('bias', entry.get('biases')))
            return W, b

        # If it's a list, use positional indexing directly
        if isinstance(weights_dict, (list, tuple)):
            for i, layer in enumerate(self.layers):
                entry = weights_dict[i]
                W, b = _extract(entry) if isinstance(entry, dict) else (entry[0], entry[1])
                if W.shape != layer.W.shape and W.T.shape == layer.W.shape:
                    W = W.T
                b = np.array(b).reshape(-1, 1) if np.array(b).ndim == 1 else b
                layer.W = W.copy(); layer.bias = b.copy()
            return

        for i, layer in enumerate(self.layers):
            W, b = None, None

            #  A: 'layer_0_W' / 'layer_0_b'  (skeleton) ---
            if f'layer_{i}_W' in weights_dict and f'layer_{i}_b' in weights_dict:
                W = weights_dict[f'layer_{i}_W']
                b = weights_dict[f'layer_{i}_b']

            # B: 'layer_0_W' / 'layer_0_bias' --- others i had to add bcoz of autograder errors
            elif f'layer_{i}_W' in weights_dict and f'layer_{i}_bias' in weights_dict:
                W = weights_dict[f'layer_{i}_W']
                b = weights_dict[f'layer_{i}_bias']

            # C: 'W_0' / 'b_0'  
            elif f'W_{i}' in weights_dict and f'b_{i}' in weights_dict:
                W = weights_dict[f'W_{i}']
                b = weights_dict[f'b_{i}']
            elif f'W{i}' in weights_dict and f'b{i}' in weights_dict:
                W = weights_dict[f'W{i}']
                b = weights_dict[f'b{i}']

            # D: string integer key  '0', '1', 
            elif str(i) in weights_dict and isinstance(weights_dict[str(i)], dict):
                W, b = _extract(weights_dict[str(i)])

            # E: 'layer_0' nested dict 
            elif f'layer_{i}' in weights_dict and isinstance(weights_dict[f'layer_{i}'], dict):
                W, b = _extract(weights_dict[f'layer_{i}'])

            # F: Python int key ---
            elif i in weights_dict and isinstance(weights_dict[i], dict):
                W, b = _extract(weights_dict[i])

            if W is None or b is None:
                raise KeyError(f"Weights for layer {i} not found "
                               f"(keys present: {list(weights_dict.keys())[:8]})")
            # Auto-transpose W if (input_size, layer_size)
            # while our layers store W as (layer_size, input_size).
            if W.shape != layer.W.shape:
                if W.T.shape == layer.W.shape:
                    W = W.T
                # No reshape fallback 

            # Normalize bias to column vector(layer_size, 1)
            b = np.array(b)
            if b.ndim == 1:
                b = b.reshape(-1, 1)
            elif b.shape == (1, layer.bias.shape[0]):
                b = b.T
            layer.W = W.copy()
            layer.bias = b.copy()
           
    def train(self, X_train, y_train, epochs, batch_size, X_val=None, y_val=None):
        """
        Train the network for specified epochs.
        """
        epoch_loss = []
        best_val_f1 = -1.0
        self.best_weights = self.get_weights() # Initialize best weights
        
        for epoch in range(epochs):
            number_of_batches = X_train.shape[1] // batch_size
            # but if the number of samples is not perfectly divisible by batch size, then we will have some remaining samples in the last batch, so we need to handle that case as well, we can simply ignore those remaining samples for simplicity, or we can create a smaller batch for those remaining samples, but for now we will ignore those remaining samples for simplicity.
            batch_losses = []
            correct_train_predictions = 0
            
            for X_batch,y_batch in batch_generator(X_train, y_train, batch_size):               
                # forward pass
                batch_pred= self.forward(X_batch)
                
                # count correct predictions for training accuracy
                pred_labels = np.argmax(batch_pred, axis=0)
                true_labels = np.argmax(y_batch, axis=0)
                correct_train_predictions += np.sum(pred_labels == true_labels)
                
            # track the loss per epoch using softmax probabilities for correct loss calculation
                softmax_probs = self.layers[-1].output
                if self.loss_type == 'cross_entropy':
                    loss = objective_functions.categorical_cross_entropy(y_batch, softmax_probs)
                    
                else: # mse
                    loss = objective_functions.mse(y_batch, softmax_probs)
                
                batch_losses.append(np.mean(loss))# mean loss of the batch 

                # backward pass
                # calls its own ability/function of backward pass
                self.backward(y_batch,batch_pred)   

                # update weights
                self.update_weights()
               
                
            mean_epoch_loss = np.mean(batch_losses)
            train_accuracy = correct_train_predictions / X_train.shape[1]
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_accuracy, val_loss, val_f1 = self.evaluate(X_val, y_val, verbose=False)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {mean_epoch_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f} - Val F1: {val_f1:.4f}")
                
                # Check and save best weights based on val F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.best_weights = self.get_weights()
                    print(f"New best validation F1 score: {best_val_f1:.4f}. Saving best weights.")
                
                # Log to Weights & Biases
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": mean_epoch_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1
                })
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {mean_epoch_loss:.4f} - Train Acc: {train_accuracy:.4f}")
                self.best_weights = self.get_weights() # If no eval set, best_weights is just the latest weights
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": mean_epoch_loss,
                    "train_accuracy": train_accuracy
                })
                
            epoch_loss.append(mean_epoch_loss)
            
        # At the end of training, restore the model weights to the ones that achieved the best F1 score
        if hasattr(self, 'best_weights'):
            print("Restoring model to best weights based on validation F1 score.")
            self.set_weights(self.best_weights)
            
        return epoch_loss
    
    def evaluate(self, X, y, verbose=True):
        """
        Evaluate the network on given data.
        """
        # X and y are of the form d*N and c*N
        output = self.forward(X)
        # Compute accuracy
        predicted_class_index = np.argmax(output, axis=0)
        true_class_index = np.argmax(y, axis=0)
        correct_predictions = np.sum(predicted_class_index == true_class_index)
        accuracy = correct_predictions / y.shape[1] 
        
        # Compute macro F1 score
        f1 = f1_score(true_class_index, predicted_class_index, average='macro')
        
        if verbose:
            print(f"Evaluation Accuracy: {accuracy}")
            print(f"Evaluation F1 Score: {f1}")

        # compute loss using softmax probabilities from the output layer
        softmax_probs = self.layers[-1].output
        if self.loss_type == 'cross_entropy':
            loss = np.mean(objective_functions.categorical_cross_entropy(y, softmax_probs))
        else:
            loss = np.mean(objective_functions.mse(y, softmax_probs))    
            
        if verbose:
            print(f"Evaluation Loss: {loss}")
        
        return accuracy, loss, f1
