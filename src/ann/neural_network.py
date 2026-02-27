"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np

from ann import optimizers
from ann.neural_layer import Layer
import objective_functions
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
        else:
            raise ValueError(f"Unsupported optimizer: {cli_args.optimizer}")

        self.input_size = 784  # for MNIST dataset
        self.output_size = 10  # for MNIST dataset

        for i in range(self.num_of_hidden_layers):
            layer_type = "hidden"
            if i==0:
                current_input_size = self.input_size
            else:
                current_input_size = self.hidden_layers_sizes[i-1] # for hidden layer i (starting from 0 index), the input size is the layer size of the i-1 th hidden layer
            
            if i==self.num_of_hidden_layers-1:
                layer_type = "output"
            hidden_layer_i = Layer(current_input_size, self.hidden_layers_sizes[i], weight_init=self.weight_init_method,layer_type=layer_type)
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

        current_input_signal = X
        for layer in self.layers:
            current_output = layer.forward(current_input_signal)
            current_input_signal = current_output
        return current_output
        # this is the final logits of the output layer (if everythying works fine, this is predicted proabalities for each class for each sample in the batch, so it should be of size (output_size, batch_size)) = y_hat
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        error_signal = y_pred - y_true # to start with the last error signal
        # we will directly use the short cut formula for the output layer, which is del = s - y, where s is the output of softmax and y is the true label in one hot encoding, so we dont need to compute the actual derivative of softmax activation function, which is computationally expensive, and we can directly use this error signal for backpropagation through the output layer and then through the hidden layers as usual using their activation function derivatives.
        # so basically the backward function needs del.w from next layer and then it calculates its del. this del is used for calcuting gradients. but for output layer, the del i.e. dL/dz is already known as error = y_pred - y-true
        # so ASSUMING WE HAVE ACTIVATION DERIVATIVE FOR OUTPUT LAYER SET AS 1, we can directly input this error signal right from the output layer, and everything will work fine.

        for layer in reversed(self.layers):
            intermed_error = layer.backward(error_signal)  # this will compute and store the graadients and return del.w value which is the input for the next layer in the reverse order
            error_signal = intermed_error
            
        # with this we have gradients stored in each layer object where each value in the matrix rep the sum of gradients of Loss wrt to that weight across the batch
        # so we might have to divide by batch size to get the mean gradient for each weight, and then we can use these gradients to update the weights using the optimizer, which will be implemented in the update_weights() method of this class.
        
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update(self.layers)
           
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        epoch_loss = []
        for epoch in range(epochs):
            number_of_batches = X_train.shape[1] // batch_size
            # but if the number of samples is not perfectly divisible by batch size, then we will have some remaining samples in the last batch, so we need to handle that case as well, we can simply ignore those remaining samples for simplicity, or we can create a smaller batch for those remaining samples, but for now we will ignore those remaining samples for simplicity.
            batch_losses = []
            
            for X_batch,y_batch in batch_generator(X_train, y_train, batch_size):               
                # forward pass
                batch_pred= self.forward(X_batch)
                
            # track the loss per epoch
                if self.loss_type == 'cross_entropy':
                    loss = objective_functions.categorical_cross_entropy(y_batch, batch_pred)
                    
                else: # mse
                    loss = objective_functions.mse(y_batch, batch_pred)
                
                batch_losses.append(np.mean(loss))# mean loss of the batch 

                # backward pass
                # calls its own ability/function of backward pass
                self.backward(y_batch,batch_pred)   

                # update weights
                self.update_weights()
               
                
            mean_epoch_loss= np.mean(batch_losses) # average loss for the epoch
            print(f"Epoch {epoch+1}, Loss: {mean_epoch_loss}")
            epoch_loss.append(mean_epoch_loss)
        return epoch_loss # this is a list of mean loss values for each epoch, which can be used for plotting the loss curve after training is done.
    
    def evaluate(self, X, y):
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
        print(f"Evaluation Accuracy: {accuracy}")

        # compute loss
        if self.loss_type == 'cross_entropy':
            loss = np.mean(objective_functions.categorical_cross_entropy(y, output))
        else:
            loss = np.mean(objective_functions.mse(y, output))    
            
        print(f"Evaluation Loss: {loss}")
        
        return accuracy, loss
